import json
import os

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# Import benchmark environments
from benchmarks.target_threat import TargetThreatEnv
from benchmarks.delayed_lure import DelayedLureEnv
from benchmarks.abstraction_unseen import AbstractionUnseenEnv
from benchmarks.negative_control import NegativeControlEnv
from benchmarks.context_flip import ContextFlipEnv
from benchmarks.self_state_meaning import SelfStateMeaningEnv
from benchmarks.triadic_binding import TriadicBindingEnv
from benchmarks.role_transformation import RoleTransformationEnv
from benchmarks.affordance_gap import AffordanceGapEnv
from benchmarks.situated_gap import SituatedGapEnv
from benchmarks.code_body import CodeBodyEnv
from benchmarks.code_body_real import CodeBodyRealEnv
from benchmarks.randomized_consequence import RandomizedConsequenceEnv
from olf.organism import Organism
from olf.baselines import MLPBaselineAgent, AblatedOrganism
from olf.seeding import set_seed
from experiments.metrics import MetricTracker


# v0.3.2.11: Core tasks for FLC ablation comparison.
# These are the primary behavioral benchmarks where FLC's contribution
# is most likely to be measurable.
FLC_TASKS = [
    "self_state_meaning",
    "delayed_lure",
    "triadic_binding",
    "target_threat",
]


class AbstractionUnseenRandomPosEnv(AbstractionUnseenEnv):
    """abstraction_unseen with randomized entity positions (leakage diagnostic)."""

    def __init__(self, seed=None):
        super().__init__(seed=seed, randomize_positions=True)


# Map benchmark names to environment classes
ENV_MAP = {
    "target_threat": TargetThreatEnv,
    "delayed_lure": DelayedLureEnv,
    "abstraction_unseen": AbstractionUnseenEnv,
    "negative_control": NegativeControlEnv,
    "context_flip": ContextFlipEnv,
    "self_state_meaning": SelfStateMeaningEnv,
    "triadic_binding": TriadicBindingEnv,
    "role_transformation": RoleTransformationEnv,
    "affordance_gap": AffordanceGapEnv,
    "situated_gap": SituatedGapEnv,
    "code_body": CodeBodyEnv,
    "code_body_real": CodeBodyRealEnv,
    "randomized_consequence": RandomizedConsequenceEnv,
    "abstraction_unseen_randompos": AbstractionUnseenRandomPosEnv,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# v0.3.1.2: feature flags. All default to off / current behavior. The
# diagnostic runner sets these explicitly per group.
FEATURE_FLAGS = {
    # Counterfactual loss (Constitution §3 FiLM + counterfactual pair).
    # v0.3.2: DISABLED by default. Was destabilizing at 0.005.
    # Enable per-task in the diagnostic runner for self_state_meaning only.
    # Stop if triadic_binding regresses below 56%.
    "counterfactual_loss": 0.0,
    # v0.3.2: regression gate — if triadic_binding drops below this
    # threshold, CF loss is disabled for the rest of the run.
    "triadic_binding_floor": 0.56,
}


def set_global_seed(seed):
    """Seed Python, NumPy, and PyTorch for fully deterministic runs.

    Per user decision: full determinism so reruns are bit-exact.
    """
    set_seed(seed)


def _tensor_health(tensor):
    """Small finite-value summary for training diagnostics."""
    with torch.no_grad():
        t = tensor.detach()
        finite = torch.isfinite(t)
        finite_values = t[finite]
        return {
            "shape": list(t.shape),
            "all_finite": bool(finite.all().item()),
            "has_nan": bool(torch.isnan(t).any().item()),
            "has_inf": bool(torch.isinf(t).any().item()),
            "norm": (
                float(torch.linalg.norm(finite_values).item())
                if finite_values.numel()
                else 0.0
            ),
            "max_abs": (
                float(finite_values.abs().max().item())
                if finite_values.numel()
                else 0.0
            ),
        }


def _boundary_proximity_target(obs, next_obs, was_lethal):
    """Self-supervised B_psi target from distance to the organism boundary.

    This is not a reward label. It only says how close the body variables are
    to collapse after the transition. Death/starvation remains the hard
    positive boundary case; otherwise risk rises as the largest homeostatic
    coordinate approaches the unit boundary.
    """
    if was_lethal:
        return 1.0
    if obs is None or next_obs is None or len(next_obs) < 4:
        return 0.0

    next_self = np.asarray(next_obs[2:4], dtype=np.float32)
    boundary_pressure = float(np.max(next_self))

    if boundary_pressure <= 0.6:
        return 0.0
    if boundary_pressure <= 0.8:
        return float(0.3 + 0.4 * ((boundary_pressure - 0.6) / 0.2))
    return float(min(1.0, 0.7 + 0.3 * ((boundary_pressure - 0.8) / 0.2)))


def _passive_boundary_target(obs):
    """Counterfactual no-action boundary target for B_psi baseline samples."""
    if obs is None or len(obs) < 4:
        return 0.0
    passive_next = np.asarray(obs[2:4], dtype=np.float32).copy()
    # BaseBenchmark passive drift: the body moves toward the viability boundary
    # even when the organism does nothing.
    passive_next[0] = np.clip(passive_next[0] + 0.02, 0.0, 1.0)
    passive_next[1] = np.clip(passive_next[1] + 0.01, 0.0, 1.0)
    boundary_pressure = float(np.max(passive_next))
    if boundary_pressure <= 0.6:
        return 0.0
    if boundary_pressure <= 0.8:
        return float(0.3 + 0.4 * ((boundary_pressure - 0.6) / 0.2))
    return float(min(1.0, 0.7 + 0.3 * ((boundary_pressure - 0.8) / 0.2)))


def _attributable_boundary_target(boundary_target, passive_target, was_lethal):
    """Action-attributable B_psi target.

    B_psi is queried as an excess-risk model during inference:
    B_psi(h, a, dh_pred) - B_psi(h, zero_action, dh_pred). Training it on
    absolute boundary proximity makes that difference collapse when passive
    drift is already risky. The target therefore labels only the risk added by
    the action relative to passive drift.
    """
    if was_lethal:
        return 1.0
    return max(0.0, float(boundary_target) - float(passive_target))


def _compute_counterfactual_loss(
    agent, episode_obs, episode_actions, episode_self_states, episode_raw_rewards
):
    """v3: empirical hinge-based counterfactual loss for FiLM self-state.

    Pairs of steps with the same task situation (object features, context)
    but different self_state and different observed reward become contrast
    pairs. We require the consequence "value" prediction to differ by at
    least the observed reward difference plus a margin.

    Returns a scalar loss tensor, or None if no contrast pair is found or
    the agent has no FiLM mechanism.
    """
    # Skip for ablations that disable self-state or memory.
    if hasattr(agent, "ablation_type") and agent.ablation_type in (
        "no_self_state", "no_memory", "no_consequence_memory"
    ):
        return None
    if not hasattr(agent, "semantics") or not hasattr(agent.semantics, "film_gen"):
        return None
    if len(episode_obs) < 2:
        return None
    if not episode_self_states or len(episode_self_states) < 2:
        return None

    device = next(agent.parameters()).device
    n = len(episode_obs)
    # Obs layout: [agent(2), self_state(2), context(2), N*(rel_pos(2)+feats(4))]
    # We use object features (positions start at index 6) and context (4:6)
    # as the situation signature.
    def situation_sig(obs):
        # context (2) + entity features (last 4N) as a coarse signature.
        return np.concatenate([obs[4:6], obs[6:]])

    def self_state_vec(obs):
        return np.asarray(obs[2:4], dtype=np.float32)

    # Build (situation, self_state, reward) per step.
    sigs = [situation_sig(o) for o in episode_obs]
    sstates = [self_state_vec(o) for o in episode_obs]
    rewards = list(episode_raw_rewards)

    # Find at most 4 contrast pairs to limit compute.
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            # Same situation (use cosine-like closeness; treat as "same" if
            # squared L2 distance is below a small threshold).
            d_sit = float(np.sum((sigs[i] - sigs[j]) ** 2))
            if d_sit > 0.05:
                continue
            d_ss = float(np.sum((sstates[i] - sstates[j]) ** 2))
            if d_ss < 0.001:
                continue  # same self_state, not a contrast
            rdiff = abs(rewards[i] - rewards[j])
            if rdiff < 0.05:
                continue
            pairs.append((i, j, rdiff))
            if len(pairs) >= 4:
                break
        if len(pairs) >= 4:
            break

    if not pairs:
        return None

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    for i, j, rdiff in pairs:
        obs_i = torch.FloatTensor(episode_obs[i]).to(device).unsqueeze(0)
        obs_j = torch.FloatTensor(episode_obs[j]).to(device).unsqueeze(0)
        act_i = torch.FloatTensor(episode_actions[i]).to(device).unsqueeze(0)
        act_j = torch.FloatTensor(episode_actions[j]).to(device).unsqueeze(0)
        # Parse to (agent_pos, self_state, context, entities_pos, entities_feats).
        if obs_i.shape[-1] == agent.obs_dim:
            _, ss_i, ctx_i, ent_pos_i, ent_feats_i = agent.parse_obs(obs_i)
            _, ss_j, ctx_j, ent_pos_j, ent_feats_j = agent.parse_obs(obs_j)
        else:
            pad = torch.zeros(1, agent.obs_dim - obs_i.shape[-1], device=device)
            obs_i = torch.cat([obs_i, pad], dim=-1)
            obs_j = torch.cat([obs_j, pad], dim=-1)
            _, ss_i, ctx_i, ent_pos_i, ent_feats_i = agent.parse_obs(obs_i)
            _, ss_j, ctx_j, ent_pos_j, ent_feats_j = agent.parse_obs(obs_j)
        # SPM trace is a forward call; use zeros as a stand-in to keep the
        # contrast focused on (object, context, self_state).
        spm_zero = torch.zeros(1, agent.latent_dim, device=device)
        cons_i = agent.semantics.counterfactual_loss(
            spm_zero, ent_pos_i, ent_feats_i, ctx_i, ss_i, act_i,
            spm_zero, ent_pos_j, ent_feats_j, ctx_j, ss_j, act_j,
            target_diff=torch.tensor(rdiff, device=device),
            margin=0.2,
        )
        total_loss = total_loss + cons_i

    return total_loss / len(pairs)

def train_agent(agent, task_name, num_episodes=300, lr=0.01, seed=None, agent_type="agent"):
    """
    Trains the OLF agent (or ablation/MLP baseline) using Policy Gradients
    augmented with triadic consequence updates and RTCM retrograde causal credit attribution.
    """
    set_global_seed(seed)
    agent.to(device)
    # Decoupled optimizers: the policy head needs a much smaller learning
    # rate than the rest of the organism, because policy gradients on
    # continuous actions are high-variance and easily destroy good behavior.
    # The semantics/veto/etc also need a small learning rate because their
    # outputs feed into the policy input — even small changes there shift
    # the policy output. This is constitutional — we still learn everything
    # end-to-end, but at appropriate rates.
    policy_params = []
    other_params = []
    for name, p in agent.named_parameters():
        if not p.requires_grad:
            continue
        if "movement_policy" in name:
            policy_params.append(p)
        else:
            other_params.append(p)
    optimizer = optim.Adam([
        {"params": policy_params, "lr": lr * 0.01},
        {"params": other_params, "lr": lr * 0.1},
    ])
    bpsi_enabled = (
        hasattr(agent, "veto")
        and not hasattr(agent, "net")
        and not (
            hasattr(agent, "ablation_type")
            and agent.ablation_type in ("no_veto_boundary", "soft_risk_only")
        )
    )
    bpsi_optimizer = (
        optim.Adam(agent.veto.parameters(), lr=1e-3)
        if bpsi_enabled else None
    )
    if bpsi_enabled:
        agent._bpsi_training_stats = []

    EnvClass = ENV_MAP[task_name]
    env = EnvClass(seed=seed)

    # Running-mean baseline over the most recent episodes.
    # For sparse-reward tasks, the per-episode mean is 0 on most episodes, so
    # subtracting it would zero out the gradient. Instead we keep a sliding
    # window of recent episode returns and use that as the baseline. This is
    # a standard REINFORCE-with-baseline technique (variance reduction, NOT
    # reward shaping — we are not changing the rewards, only centering the
    # policy gradient signal around the running mean of recent returns).
    _baseline_window = 50
    _return_history: list = []
    
    rewards_history = []
    success_history = []
    
    for episode in range(num_episodes):
        # Update warmup state at start of each episode
        if hasattr(agent, 'episode_count'):
            agent.episode_count = episode
        
        obs = env.reset()
        agent.reset_state()
        
        episode_obs = []
        episode_actions = []
        episode_rewards = []
        episode_raw_rewards = []
        episode_dones = []
        episode_infos = []
        episode_self_states = []
        # v0.3.2.11: self-supervised boundary targets for B_psi.
        episode_boundary_targets = []
        episode_zero_boundary_targets = []
        # Accumulated B_psi training data (h, a, dh_pred, target).
        episode_bpsi_data = []
        
        done = False
        
        while not done:
            episode_obs.append(obs.copy())
            
            # Step forward
            action, info_dict = agent.select_action(obs)
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            
            episode_actions.append(action.copy())
            episode_raw_rewards.append(reward)
            
            # Internal consequence-driven reward shaping (Constitution §7, §19 compliant)
            # Instead of external distance-based shaping, use the organism's own
            # consequence predictions as an intrinsic motivation signal.
            # This keeps reward shaping within the organismic loop.
            intrinsic_bonus = 0.0
            if hasattr(agent, 'semantics') and hasattr(info_dict, '__getitem__'):
                cons = info_dict.get("consequences", None)
                if cons is not None and "value" in cons:
                    # Small bonus from the organism's own predicted value
                    # This decays naturally as the predictor learns real consequences
                    pred_val = cons["value"].mean().item()
                    intrinsic_bonus = 0.02 * pred_val
            
            shaped_reward = reward + intrinsic_bonus
            episode_rewards.append(shaped_reward)
            episode_dones.append(done)
            episode_infos.append(info)
            episode_self_states.append(obs[2:4].copy())
            
            # Consequence feedback to the fast/slow memory trace
            was_lethal = 1.0 if info["status"] in ["death", "starvation"] else 0.0
            
            # Calculate changes in homeostatic self state
            hunger_delta = next_obs[2] - obs[2]
            fatigue_delta = next_obs[3] - obs[3]

            # v0.3.2.11: boundary proximity provides a continuous signal before
            # actual death. The paired zero-action target teaches B_psi the
            # baseline risk of inaction at the same body state.
            episode_boundary_targets.append(
                _boundary_proximity_target(obs, next_obs, was_lethal)
            )
            episode_zero_boundary_targets.append(_passive_boundary_target(obs))
            
            agent.learn_consequence(reward, was_lethal, hunger_delta, fatigue_delta)
            
            obs = next_obs
            
        # 1. Compute return G_t for policy gradient calculation
        returns = []
        G = 0.0
        for r in reversed(episode_rewards):
            G = r + 0.95 * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)

        # Update running-mean baseline. Use the sum of episode raw rewards
        # (not discounted return) so the baseline tracks "average episode
        # outcome", which is what we want to center the policy gradient around.
        ep_total = sum(episode_raw_rewards)
        _return_history.append(ep_total)
        if len(_return_history) > _baseline_window:
            _return_history.pop(0)
        # Running mean: average of recent episode returns.
        running_mean = sum(_return_history) / len(_return_history)
        # Per-episode baseline for this episode.
        baseline = torch.tensor(running_mean, device=device)
        advantages = returns - baseline

        # 2. Retrieve RTCM retrograde causal blame if a delayed reward or hazard occurs
        # The consequence vector: [final_reward, was_lethal, hunger_delta, fatigue_delta]
        final_info = episode_infos[-1]
        final_was_lethal = 1.0 if final_info["status"] in ["death", "starvation"] else 0.0
        final_reward = sum(episode_rewards)

        consequence_vec = torch.FloatTensor([[final_reward, final_was_lethal, 0.0, 0.0]]).to(device)

        # Retrograde blame weights: scale gradient updates of historical steps based on causal blame
        if hasattr(agent, 'rtcm'):
            blame_weights = agent.rtcm.retrieve_causal_blame(consequence_vec)
            if not blame_weights:
                blame_weights = [1.0] * len(episode_obs)
        else:
            blame_weights = [1.0] * len(episode_obs)

        # v3: blend in RTCM delayed-credit blame (top-k softmax retrieval).
        # The delayed-credit weights point at the top-k past steps most likely
        # to have caused the observed consequence. We blend 50/50 with the
        # existing per-step blame so the organism still uses per-step signal
        # but adds delayed causal retrieval. If RTCM returns nothing, the
        # blend leaves the original blame untouched.
        if hasattr(agent, 'rtcm') and not (
            hasattr(agent, "ablation_type") and agent.ablation_type == "no_rtcm"
        ):
            delayed = agent.rtcm.retrieve_delayed_credit(
                consequence_vec, agent.h, top_k=3
            )
            if delayed and len(delayed) == len(episode_obs):
                # Build a full-length per-step delayed weight vector (top-k
                # entries filled, rest zero).
                delayed_full = [0.0] * len(episode_obs)
                for idx, w in delayed:
                    if 0 <= idx < len(delayed_full):
                        delayed_full[idx] += w
                # 50/50 blend with existing blame (v3 amendment: blend = 0.5
                # per locked spec). The result is a per-step scaling; the
                # existing training loop multiplies policy_loss by
                # causal_weight = blame_weights[idx] * ret * adv.
                if len(blame_weights) == len(delayed_full):
                    blame_weights = [
                        0.5 * b + 0.5 * d
                        for b, d in zip(blame_weights, delayed_full, strict=False)
                    ]

            # v3 (Action-Sphere RTCM memo §3): transfer-aware retrieval.
            # "Inverse transfer produces a cause-space query that can locate
            # old causes far better than raw effect similarity." Use this
            # AS the cause-space query (not just reranking) by blending its
            # weights with the existing blame at a smaller weight (0.3) so
            # the per-step signal remains dominant.
            # v3.1: temporarily disabled — was destabilizing learning on
            # triadic_binding. We rely on the simpler retrieve_delayed_credit.
            # if hasattr(agent.rtcm, "transfer_aware_retrieve"):
            #     ta = agent.rtcm.transfer_aware_retrieve(
            #         consequence_vec, agent.h, top_k=5
            #     )
            #     if ta and len(ta) == len(episode_obs):
            #         ta_full = [0.0] * len(episode_obs)
            #         for idx, w in ta:
            #             if 0 <= idx < len(ta_full):
            #                 ta_full[idx] += w
            #         if len(blame_weights) == len(ta_full):
            #             blame_weights = [
            #                 0.7 * b + 0.3 * t
            #                 for b, t in zip(blame_weights, ta_full)
            #             ]

        # 3. Optimize parameters
        optimizer.zero_grad()
        agent.reset_state()
        
        policy_loss = 0.0
        consequence_loss = 0.0
        
        if hasattr(agent, 'net'):
            # MLP agent training loop
            for idx in range(len(episode_obs)):
                obs_step = episode_obs[idx]
                act_taken = torch.FloatTensor(episode_actions[idx]).to(device).unsqueeze(0)
                adv = advantages[idx]
                obs_t = torch.FloatTensor(obs_step).to(device).unsqueeze(0)

                pred_act = agent.net(obs_t)
                dist_dx = torch.distributions.Normal(pred_act[:, 0], 0.1)
                dist_dy = torch.distributions.Normal(pred_act[:, 1], 0.1)
                dist_u = torch.distributions.Normal(pred_act[:, 2], 0.1)
                log_prob = dist_dx.log_prob(act_taken[:, 0]) + dist_dy.log_prob(act_taken[:, 1]) + dist_u.log_prob(act_taken[:, 2])
                policy_loss += -log_prob.sum() * adv
        else:
            # OLF Agent training loop
            for idx in range(len(episode_obs)):
                obs_step = episode_obs[idx]
                act_taken = torch.FloatTensor(episode_actions[idx]).to(device).unsqueeze(0)
                adv = advantages[idx]
                causal_weight = blame_weights[idx] if idx < len(blame_weights) else 1.0
                obs_t = torch.FloatTensor(obs_step).to(device).unsqueeze(0)

                # Parse observations (organism.parse_obs now returns 5 values)
                if obs_t.shape[-1] == agent.obs_dim:
                    agent_pos, self_state, context, entities_pos, entities_feats = agent.parse_obs(obs_t)
                else:
                    pad = torch.zeros(1, agent.obs_dim - obs_t.shape[-1], device=device)
                    obs_t = torch.cat([obs_t, pad], dim=-1)
                    agent_pos, self_state, context, entities_pos, entities_feats = agent.parse_obs(obs_t)

                # Retrieve trace
                if hasattr(agent, 'spm'):
                    spm_trace = agent.spm.get_trace().to(device)
                else:
                    spm_trace = torch.zeros(1, agent.latent_dim, device=device)

                # Update latent flow proposal
                x_flow = torch.cat([obs_t, spm_trace], dim=-1)
                agent.h = agent.flow(x_flow, agent.h)

                # entities_pos and entities_feats already come from parse_obs.

                # Bind situated representation
                sigma_t = agent.semantics.bind(spm_trace, entities_pos, entities_feats, context, self_state)

                # REINFORCE update for movement policy network
                if hasattr(agent, 'movement_policy'):
                    flat_embeds = sigma_t.reshape(1, -1)
                    policy_inputs = torch.cat([agent.h, flat_embeds], dim=-1)
                    pred_move = agent.movement_policy(policy_inputs)
                    if hasattr(agent, "apply_future_control"):
                        pred_move, _ = agent.apply_future_control(
                            sigma_t, self_state, pred_move
                        )

                    # Non-finite diagnostics: if any policy output is non-finite,
                    # crash loudly with full context so the source is identifiable.
                    # Do NOT silently continue: non-finite policy outputs indicate
                    # a real numerical instability that must be diagnosed.
                    if not torch.isfinite(pred_move).all():
                        _ctx = {
                            "task": task_name,
                            "seed": seed,
                            "agent_type": agent_type,
                            "ablation_type": getattr(agent, "ablation_type", None),
                            "episode": episode,
                            "step": idx,
                            "h": _tensor_health(agent.h),
                            "spm_trace": _tensor_health(spm_trace),
                            "sigma_t": _tensor_health(sigma_t),
                            "policy_inputs": _tensor_health(policy_inputs),
                            "pred_move_health": _tensor_health(pred_move),
                            "pred_move": pred_move.detach().tolist(),
                        }
                        raise RuntimeError(
                            f"[non-finite diagnostics] Non-finite pred_move at "
                            f"episode={episode} step={idx}  {_ctx}"
                        )

                    # 3-dim policy: dx, dy, u (use action).
                    # Continuous Normal likelihood for variance reduction even on the
                    # binary use action.
                    dist_dx = torch.distributions.Normal(pred_move[:, 0], 0.1)
                    dist_dy = torch.distributions.Normal(pred_move[:, 1], 0.1)
                    dist_u = torch.distributions.Normal(pred_move[:, 2], 0.1)
                    log_prob = (dist_dx.log_prob(act_taken[:, 0])
                                + dist_dy.log_prob(act_taken[:, 1])
                                + dist_u.log_prob(act_taken[:, 2]))

                    # Scale policy gradient loss by the causal blame weight from RTCM
                    # Use advantage (centered return) instead of raw return for variance reduction.
                    policy_loss += -log_prob.sum() * adv * causal_weight
                    
                    # Train semantics consequence predictions
                    # SCALED DOWN: most episodes end with reward=0, so training the
                    # semantics to "predict 0" drowns out the policy gradient with
                    # gradients that are misaligned with the actual task. We only
                    # train the consequence predictor on the last step of the
                    # episode (where reward/lethality carry actual signal) and at
                    # a small weight to avoid overwhelming the policy gradient.
                    if idx == len(episode_obs) - 1:
                        next_info = episode_infos[idx]
                        was_lethal_step = 1.0 if next_info["status"] in ["death", "starvation"] else 0.0
                        reward_val = episode_rewards[idx]

                        for ent_idx in range(entities_pos.size(1)):
                            ent_dist = torch.linalg.norm(entities_pos[0, ent_idx]).item()
                            if ent_dist < 0.15:
                                # Consequence mapping target: [reward, was_lethal, hunger_delta, fatigue_delta]
                                target_cons = torch.FloatTensor([reward_val, was_lethal_step, 0.0, 0.0]).to(device)

                                # Forward prediction
                                consequences_pred = agent.semantics.predict_consequences(sigma_t, act_taken)
                                consequence_loss += nn.MSELoss()(consequences_pred["value"][0, ent_idx], target_cons[0:1])
                                consequence_loss += nn.MSELoss()(consequences_pred["terminal_risk"][0, ent_idx], target_cons[1:2])

                    # v0.3.2.11: Accumulate B_psi training data on EVERY step.
                    # B_psi learns ACTION-ATTRIBUTABLE boundary risk: how much
                    # does this action increase risk relative to passive drift?
                    #
                    # Target for action sample:
                    #   attributable_risk = max(0, proximity_with_action - proximity_passive)
                    #   This is > 0 only when the action makes things worse than doing nothing.
                    #   Death always has attributable_risk = 1.0.
                    #
                    # Target for zero-action sample:
                    #   Always 0.0: zero action has zero attributable risk by definition.
                    #
                    # At inference: danger = B_psi(a) - B_psi(0) ~= B_psi(a),
                    # which is non-zero only when the action increases boundary risk.
                    if bpsi_enabled:
                        consequences_veto = agent.semantics.predict_consequences(sigma_t, act_taken)
                        dh_pred_veto = consequences_veto["dh_pred"].mean(dim=1)

                        was_lethal_step = (
                            1.0 if episode_infos[idx]["status"] in ["death", "starvation"]
                            else 0.0
                        )
                        boundary_target = (
                            episode_boundary_targets[idx]
                            if idx < len(episode_boundary_targets) else 0.0
                        )
                        zero_target = (
                            episode_zero_boundary_targets[idx]
                            if idx < len(episode_zero_boundary_targets) else 0.0
                        )

                        # Compute attributable risk: how much worse is the action
                        # compared to passive drift?
                        attributable_risk = _attributable_boundary_target(
                            boundary_target, zero_target, was_lethal_step
                        )

                        action_weight = 1.0
                        if was_lethal_step:
                            action_weight = 12.0
                        elif attributable_risk > 0.05:
                            action_weight = 4.0

                        episode_bpsi_data.append((
                            agent.h.clone().detach(),
                            act_taken.clone().detach(),
                            dh_pred_veto.clone().detach(),
                            torch.tensor([[attributable_risk]], device=device),
                            torch.tensor([[action_weight]], device=device),
                        ))

                        # Zero-action sample: target is always 0.0
                        a_zero = torch.zeros_like(act_taken)
                        consequences_zero = agent.semantics.predict_consequences(sigma_t, a_zero)
                        dh_pred_zero = consequences_zero["dh_pred"].mean(dim=1)
                        episode_bpsi_data.append((
                            agent.h.clone().detach(),
                            a_zero.clone().detach(),
                            dh_pred_zero.clone().detach(),
                            torch.tensor([[0.0]], device=device),
                            torch.tensor([[1.0]], device=device),
                        ))

                    # Update SPM trace buffer during training replay (Constitution compliance)
                    if hasattr(agent, 'spm'):
                        if not (hasattr(agent, 'ablation_type') and agent.ablation_type == 'no_diagnostic_decay'):
                            agent.spm.update(agent.h)

        # v3: empirical counterfactual loss for FiLM self-state modulation.
        # Find pairs of steps in the same episode where:
        #   - object features and context are similar (same task situation)
        #   - self_state differs
        #   - observed reward differs
        # Then require the consequence "value" prediction to differ.
        # Hinge-based, only empirical contrast pairs (no hand-crafted labels).
        # v3.1: temporarily disabled — was destabilizing training.
        # v0.3.1.2: re-enabled at the small weight in FEATURE_FLAGS for the
        # self_state_meaning diagnostic. Default 0.0 keeps current behavior.
        total_loss = policy_loss
        if FEATURE_FLAGS["counterfactual_loss"] > 0.0:
            cf_loss = _compute_counterfactual_loss(
                agent, episode_obs, episode_actions,
                episode_self_states, episode_raw_rewards
            )
            if cf_loss is not None and cf_loss.requires_grad:
                total_loss = (
                    total_loss
                    + FEATURE_FLAGS["counterfactual_loss"] * cf_loss
                )

        # Auxiliary losses are scaled down (0.01) to avoid overwhelming the
        # policy gradient. The semantics is still being trained, but at a
        # smaller weight so it doesn't dominate the optimizer.
        # v0.3.2.10: B_psi is NO LONGER trained in the main loss.
        # It gets a dedicated training step below (like RTCM).
        if consequence_loss > 0:
            total_loss = total_loss + 0.01 * consequence_loss
            
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
            optimizer.step()

        # v0.3.2.11: Dedicated B_psi boundary-risk training step.
        # B_psi is trained SEPARATELY from the policy gradient to avoid
        # being drowned by it. Uses accumulated (h, a, dh_pred, target)
        # tuples from the episode. The optimizer is persistent across
        # episodes so Adam keeps its state.
        if bpsi_optimizer is not None and episode_bpsi_data:
            bpsi_loss_value = 0.0
            for _ in range(10):  # 10 gradient steps per episode
                bpsi_optimizer.zero_grad()
                bpsi_total_loss = torch.tensor(0.0, device=device)
                for h_t, a_t, dh_t, target_t, weight_t in episode_bpsi_data:
                    pred_logits = agent.veto.predict_risk_logits(h_t, a_t, dh_t)
                    sample_loss = nn.functional.binary_cross_entropy_with_logits(
                        pred_logits, target_t, reduction="none"
                    )
                    bpsi_total_loss = bpsi_total_loss + (sample_loss * weight_t).mean()
                bpsi_total_loss = bpsi_total_loss / len(episode_bpsi_data)
                bpsi_total_loss.backward()
                nn.utils.clip_grad_norm_(agent.veto.parameters(), 0.5)
                bpsi_optimizer.step()
                bpsi_loss_value = float(bpsi_total_loss.detach().item())

            targets = [float(target_t.detach().item()) for _, _, _, target_t, _ in episode_bpsi_data]
            weights = [float(weight_t.detach().item()) for _, _, _, _, weight_t in episode_bpsi_data]
            agent._bpsi_training_stats.append({
                "episode": episode,
                "n": len(targets),
                "attributable_mean": float(np.mean(targets)) if targets else 0.0,
                "attributable_max": float(np.max(targets)) if targets else 0.0,
                "attributable_nonzero_rate": float(np.mean([t > 0.0 for t in targets])) if targets else 0.0,
                "weight_mean": float(np.mean(weights)) if weights else 0.0,
                "loss": bpsi_loss_value,
            })

        # Constitution §15: RTCM slow-learning step.
        # Train R_Δ (cause→effect) and the blame estimator from the episode's
        # full transition history. This is a separate, slow gradient step that
        # does NOT flow through the main optimizer.
        if hasattr(agent, "rtcm") and not (
            hasattr(agent, "ablation_type")
            and agent.ablation_type in ("no_rtcm", "no_memory")
        ):
            agent.rtcm.train_step(lr=1e-3)

        # Constitution §4: slow plasticity of the flow field via ConsequenceMemory.
        # Uses stored (s_t, a_t, s_{t+1}) transitions to gently deform the LTC
        # flow proposal toward observed trajectories. lr is intentionally tiny
        # (≪ main learning rate) so fast trace memory remains dominant.
        if hasattr(agent, "consequence_memory") and hasattr(agent, "flow") and not (
            hasattr(agent, "ablation_type")
            and agent.ablation_type in ("no_memory", "no_consequence_memory")
        ):
            agent.consequence_memory.consolidate(agent.flow, lr=1e-3)

        # Log episode metrics for learning curves (Constitution §17)
        final_info = episode_infos[-1]
        is_success = 1.0 if final_info["status"] == "success" else 0.0
        raw_rewards_sum = sum(episode_raw_rewards)
        rewards_history.append(float(raw_rewards_sum))
        success_history.append(is_success)

        # v0.3.2: regression gate. If training on triadic_binding and
        # the running success rate drops below the floor, disable CF loss
        # for the rest of this run to preserve the constitutional win.
        if (task_name == "triadic_binding"
            and len(success_history) >= 20
            and FEATURE_FLAGS["counterfactual_loss"] > 0.0):
            recent_success = sum(success_history[-20:]) / 20.0
            if recent_success < FEATURE_FLAGS.get("triadic_binding_floor", 0.56):
                FEATURE_FLAGS["counterfactual_loss"] = 0.0

        # v0.3.2.5: periodic self_state swap probe (every 50 episodes)
        if hasattr(agent, "diag_mode") and agent.diag_mode and (episode + 1) % 50 == 0:
            probe = agent.self_state_swap_probe(obs)
            if not hasattr(agent, "_swap_probes"):
                agent._swap_probes = []
            probe["episode"] = episode + 1
            agent._swap_probes.append(probe)
        
    # Save learning curve history (Constitution §17)
    if seed is not None:
        os.makedirs("experiments/learning_curves", exist_ok=True)
        curve_file = f"experiments/learning_curves/{task_name}_{agent_type}_seed{seed}.json"
        with open(curve_file, "w") as f:
            json.dump({"rewards": rewards_history, "success": success_history}, f)
            
    return agent

def evaluate_agent(agent, task_name, num_episodes=50, seed=None):
    """
    Evaluates agent on the specified task and logs metrics.
    """
    set_global_seed(seed)
    agent.eval()
    EnvClass = ENV_MAP[task_name]
    env = EnvClass(seed=seed)
    tracker = MetricTracker()
    
    for _episode in range(num_episodes):
        obs = env.reset()
        agent.reset_state()
        
        done = False
        ep_reward = 0.0
        steps = 0
        inventions = 0
        
        while not done:
            action, info_dict = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1
            
            if info_dict["mode"] == 2:
                inventions += 1
                
            # Feed consequence back to the agent for online homeostatic adaptation/recoupling
            was_lethal = 1.0 if info["status"] in ["death", "starvation"] else 0.0
            hunger_delta = next_obs[2] - obs[2]
            fatigue_delta = next_obs[3] - obs[3]
            agent.learn_consequence(reward, was_lethal, hunger_delta, fatigue_delta)
            
            obs = next_obs
            
        tracker.log_run(ep_reward, info["status"], inventions, steps)
        
    return tracker.get_stats()

def run_task_experiment(task, ablations):
    """
    Worker function to run a single task's training and evaluations.
    Restricts PyTorch internal threading to 1 to avoid thread oversubscription.
    """
    torch.set_num_threads(1)
    seeds = [42, 43, 44]
    
    olf_seeds_stats = []
    mlp_seeds_stats = []
    ablation_seeds_stats = {abl: [] for abl in ablations}
    
    for seed in seeds:
        # 1. Full OLF Organism
        set_seed(seed)
        olf = Organism(obs_dim=18, action_dim=3)
        olf = train_agent(olf, task, num_episodes=300, seed=seed, agent_type="olf")
        olf_stats = evaluate_agent(olf, task, seed=seed+100)
        olf_seeds_stats.append(olf_stats)
        
        # 2. MLP RL Baseline
        set_seed(seed)
        mlp = MLPBaselineAgent(obs_dim=18, action_dim=3)
        mlp = train_agent(mlp, task, num_episodes=300, seed=seed, agent_type="mlp")
        mlp_stats = evaluate_agent(mlp, task, seed=seed+100)
        mlp_seeds_stats.append(mlp_stats)
        
        # 3. Direct Target Ablations
        for abl_type in ablations:
            set_seed(seed)
            abl_agent = AblatedOrganism(obs_dim=18, action_dim=3, ablation_type=abl_type)
            abl_agent = train_agent(abl_agent, task, num_episodes=300, seed=seed, agent_type=f"abl_{abl_type}")
            abl_stats = evaluate_agent(abl_agent, task, seed=seed+100)
            ablation_seeds_stats[abl_type].append(abl_stats)
            
    # Compute seed-averaged stats
    def average_stats(stats_list):
        avg_stats = {}
        for key in stats_list[0].keys():
            avg_stats[key] = float(np.mean([s[key] for s in stats_list]))
        return avg_stats

    olf_stats = average_stats(olf_seeds_stats)
    mlp_stats = average_stats(mlp_seeds_stats)
    ablation_results = {abl: average_stats(ablation_seeds_stats[abl]) for abl in ablations}
        
    return task, {
        "OLF": olf_stats,
        "MLP": mlp_stats,
        "Ablations": ablation_results
    }

def _evaluate_with_diagnostics(agent, task_name, seed, num_episodes=20):
    """Evaluate an agent with diagnostics enabled, returning per-episode metrics."""
    set_global_seed(seed)
    agent.eval()
    agent.diag_mode = True
    EnvClass = ENV_MAP[task_name]
    env = EnvClass(seed=seed)

    episodes = []
    for _ in range(num_episodes):
        obs = env.reset()
        agent.reset_state()
        if hasattr(agent, "reset_diag"):
            agent.reset_diag()

        done = False
        info = {"status": "running"}
        verdicts = []
        dangers = []
        while not done:
            action, action_info = agent.select_action(obs, evaluate=True)
            next_obs, reward, done, info = env.step(action)
            verdicts.append(action_info.get("verdict", "release"))
            dangers.append(float(action_info.get("danger", 0.0)))
            was_lethal = 1.0 if info["status"] in ("death", "starvation") else 0.0
            hunger_delta = next_obs[2] - obs[2]
            fatigue_delta = next_obs[3] - obs[3]
            agent.learn_consequence(reward, was_lethal, hunger_delta, fatigue_delta)
            obs = next_obs

        episodes.append({
            "status": info["status"],
            "verdict": verdicts[-1] if verdicts else "release",
            "rollback_seen": any(v == "rollback" for v in verdicts),
            "mean_danger": float(np.mean(dangers)) if dangers else 0.0,
            "max_danger": float(np.max(dangers)) if dangers else 0.0,
        })

    agent.diag_mode = False
    return episodes


def run_flc_ablation(seeds=None, num_train=300, num_eval=20):
    """Run FLC ablation suite: OLF vs no_future_latent across core tasks.

    Returns a dict with per-task results comparing full OLF to the
    no_future_latent ablation, including success rate, danger, rollback
    rate, and seed variance.
    """
    if seeds is None:
        seeds = [42, 43, 44]

    results = {}
    for task in FLC_TASKS:
        print(f"\n--- FLC ablation: {task} ---")
        per_condition = {}

        for condition, ablation_type in [
            ("olf", None),
            ("no_future_latent", "no_future_latent"),
        ]:
            seed_stats = []
            for seed in seeds:
                set_global_seed(seed)
                if ablation_type is None:
                    agent = Organism(obs_dim=18, action_dim=3)
                else:
                    agent = AblatedOrganism(
                        obs_dim=18, action_dim=3, ablation_type=ablation_type
                    )
                agent = train_agent(agent, task, num_episodes=num_train, seed=seed)

                eval_episodes = _evaluate_with_diagnostics(
                    agent, task, seed + 100, num_eval
                )

                successes = sum(
                    1 for ep in eval_episodes if ep["status"] == "success"
                )
                deaths = sum(
                    1 for ep in eval_episodes
                    if ep["status"] in ("death", "starvation")
                )
                mean_dangers = [ep["mean_danger"] for ep in eval_episodes]
                max_dangers = [ep["max_danger"] for ep in eval_episodes]
                rollbacks = sum(
                    1 for ep in eval_episodes if ep["rollback_seen"]
                )

                seed_stats.append({
                    "seed": seed,
                    "success_rate": successes / max(1, len(eval_episodes)),
                    "safety_rate": 1.0 - deaths / max(1, len(eval_episodes)),
                    "mean_danger": (
                        float(np.mean(mean_dangers)) if mean_dangers else 0.0
                    ),
                    "max_danger": (
                        float(np.max(max_dangers)) if max_dangers else 0.0
                    ),
                    "rollback_rate": rollbacks / max(1, len(eval_episodes)),
                    "total_episodes": len(eval_episodes),
                })

            per_condition[condition] = {
                "per_seed": seed_stats,
                "success_mean": float(np.mean([s["success_rate"] for s in seed_stats])),
                "success_std": float(np.std([s["success_rate"] for s in seed_stats])),
                "safety_mean": float(np.mean([s["safety_rate"] for s in seed_stats])),
                "safety_std": float(np.std([s["safety_rate"] for s in seed_stats])),
                "danger_mean": float(np.mean([s["mean_danger"] for s in seed_stats])),
                "danger_std": float(np.std([s["mean_danger"] for s in seed_stats])),
                "rollback_mean": float(np.mean([s["rollback_rate"] for s in seed_stats])),
                "rollback_std": float(np.std([s["rollback_rate"] for s in seed_stats])),
            }

        # Summary delta: how much does FLC change the outcome?
        olf = per_condition["olf"]
        nfl = per_condition["no_future_latent"]
        delta = {
            "success_delta": olf["success_mean"] - nfl["success_mean"],
            "danger_delta": olf["danger_mean"] - nfl["danger_mean"],
            "rollback_delta": olf["rollback_mean"] - nfl["rollback_mean"],
        }

        results[task] = {
            "conditions": per_condition,
            "delta": delta,
        }

        print(
            f"  OLF: success={olf['success_mean']:.1%} ± {olf['success_std']:.1%}, "
            f"danger={olf['danger_mean']:.4f}, rollback={olf['rollback_mean']:.1%}"
        )
        print(
            f"  NFL: success={nfl['success_mean']:.1%} ± {nfl['success_std']:.1%}, "
            f"danger={nfl['danger_mean']:.4f}, rollback={nfl['rollback_mean']:.1%}"
        )
        print(
            f"  delta: success={delta['success_delta']:+.1%}, "
            f"danger={delta['danger_delta']:+.4f}, rollback={delta['rollback_delta']:+.1%}"
        )

    return results


def main():
    print("=" * 110)
    print("      ORGANISMIC LATENT FLOW (OLF) THEORETICAL ABLATION SUITE RUNNER (PARALLEL)")
    print("=" * 110)
    
    # Task list and target ablations mapping
    experiments = [
        ("target_threat", ["no_veto_boundary", "soft_risk_only"]),
        ("delayed_lure", ["no_spm", "last_observation_only", "no_rtcm", "no_memory"]),
        ("abstraction_unseen", ["no_consequence_memory", "exact_episodic_memory_only", "no_abstraction"]),
        ("abstraction_unseen_randompos", []),  # v0.3.2: leakage diagnostic
        ("negative_control", []),
        ("randomized_consequence", []),  # Constitution §16 anti-cheat control
        ("context_flip", ["last_observation_only"]),
        ("self_state_meaning", ["no_self_state"]),
        ("triadic_binding", ["no_self_state", "no_spm", "no_memory"]),
        ("role_transformation", ["no_mode_arbitration", "no_closure_pressure"]),
        ("affordance_gap", ["no_invention", "ungated_invention"]),
        ("situated_gap", ["no_invention", "no_situation"]),
        ("code_body", ["no_recoupling_constraint", "no_diagnostic_decay", "inspect_only_trace"])
    ]
    
    suite_results = {}
    
    import concurrent.futures
    print("Spawning parallel workers for benchmarks (multicore)...")
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(run_task_experiment, task, ablations): task
            for task, ablations in experiments
        }
        
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                _, result = future.result()
                print(f"\nCompleted Task: {task.upper()}")
                print(f"  [Full OLF] Success: {result['OLF']['success_rate']:.1%}, Safety: {result['OLF']['safety_rate']:.1%}")
                print(f"  [MLP Baseline] Success: {result['MLP']['success_rate']:.1%}, Safety: {result['MLP']['safety_rate']:.1%}")
                for abl_type, stats in result["Ablations"].items():
                    print(f"  [{abl_type}] Success: {stats['success_rate']:.1%}, Safety: {stats['safety_rate']:.1%}")
                suite_results[task] = result
            except Exception as exc:
                print(f"Task {task} generated an exception: {exc}")
                
    # Print the gorgeous comparison matrix
    print("\n" + "=" * 115)
    print(" " * 35 + "OLF COMPARATIVE ABLATION MATRIX")
    print("=" * 115)
    print(f"{'Benchmark Task':<25} | {'Full OLF':<16} | {'MLP RL':<16} | {'Ablated Variants':<50}")
    print("-" * 115)
    
    # Order results to match original list order
    for task, _ in experiments:
        if task not in suite_results:
            continue
        data = suite_results[task]
        olf_str = f"{data['OLF']['success_rate']:.1%} ({data['OLF']['safety_rate']:.1%})"
        mlp_str = f"{data['MLP']['success_rate']:.1%} ({data['MLP']['safety_rate']:.1%})"
        
        abl_strings = []
        for name, stats in data["Ablations"].items():
            abl_strings.append(f"{name}: {stats['success_rate']:.0%} ({stats['safety_rate']:.0%})")
            
        abl_str = " | ".join(abl_strings) if abl_strings else "None Required"
        
        print(f"{task:<25} | {olf_str:<16} | {mlp_str:<16} | {abl_str:<50}")
        
    print("=" * 115)

    # --- FLC Ablation Suite ---
    print("\n" + "=" * 115)
    print(" " * 30 + "FLC ABLATION SUITE (OLF vs no_future_latent)")
    print("=" * 115)

    flc_results = run_flc_ablation()

    print("\n" + "=" * 115)
    print(f"{'Task':<25} | {'OLF success':<18} | {'NFL success':<18} | {'Δ success':<12} | {'Δ danger':<12} | {'Δ rollback':<12}")
    print("-" * 115)
    for task in FLC_TASKS:
        if task not in flc_results:
            continue
        r = flc_results[task]
        olf = r["conditions"]["olf"]
        nfl = r["conditions"]["no_future_latent"]
        d = r["delta"]
        print(
            f"{task:<25} | "
            f"{olf['success_mean']:.1%} ± {olf['success_std']:.1%}   | "
            f"{nfl['success_mean']:.1%} ± {nfl['success_std']:.1%}   | "
            f"{d['success_delta']:+.1%}       | "
            f"{d['danger_delta']:+.4f}     | "
            f"{d['rollback_delta']:+.1%}"
        )
    print("=" * 115)

    # Save FLC results to ignored output path
    os.makedirs("results/flc_ablation", exist_ok=True)
    with open("results/flc_ablation/results.json", "w") as f:
        json.dump(flc_results, f, indent=2)
    print("\nFLC ablation results saved to results/flc_ablation/results.json")


if __name__ == "__main__":
    main()

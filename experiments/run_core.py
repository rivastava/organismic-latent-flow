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


TRAINING_SIGNALS = (
    "legacy_reward",
    "raw_reward",
    "terminal_viability",
    "homeostatic_delta",
    "terminal_homeostasis",
)
CREDIT_MODES = ("uniform", "rtcm")
OPTIMIZER_PROFILES = ("legacy", "policy_focused", "uniform_low")


def _policy_learning_signal(
    raw_reward,
    *,
    done,
    was_lethal,
    predicted_value=0.0,
    training_signal="legacy_reward",
    self_state=None,
    next_self_state=None,
):
    """Return the scalar that trains policy and consequence pathways.

    ``legacy_reward`` preserves the historical runner exactly: benchmark reward
    plus the small predicted-value bonus. ``raw_reward`` removes that circular
    bonus while retaining benchmark reward. ``terminal_viability`` uses only
    terminal life/death. ``homeostatic_delta`` uses the observed deformation
    of the body viability functional. ``terminal_homeostasis`` uses its
    absolute terminal value. Neither homeostatic mode reads benchmark reward.
    """
    if training_signal not in TRAINING_SIGNALS:
        raise ValueError(
            f"unknown training_signal={training_signal!r}; expected one of {TRAINING_SIGNALS}"
        )
    if training_signal == "legacy_reward":
        return float(raw_reward) + 0.02 * float(predicted_value)
    if training_signal == "raw_reward":
        return float(raw_reward)
    if training_signal == "terminal_viability":
        if not done:
            return 0.0
        return -1.0 if was_lethal else 1.0

    if training_signal == "terminal_homeostasis":
        if not done:
            return 0.0
        if was_lethal:
            return -1.0
        if next_self_state is None:
            raise ValueError(
                "terminal_homeostasis requires next self_state"
            )
        following = np.asarray(next_self_state, dtype=np.float32)
        return float(np.clip(1.0 - following.sum(), -1.0, 1.0))

    if self_state is None or next_self_state is None:
        raise ValueError(
            "homeostatic_delta requires current and next self_state"
        )
    current = np.asarray(self_state, dtype=np.float32)
    following = np.asarray(next_self_state, dtype=np.float32)
    current_viability = float(np.clip(1.0 - current.sum(), -1.0, 1.0))
    next_viability = (
        -1.0
        if was_lethal
        else float(np.clip(1.0 - following.sum(), -1.0, 1.0))
    )
    return next_viability - current_viability


def _mean_one_credit_weights(weights, length):
    """Return non-negative credit weights with unit mean.

    Causal attribution may redistribute a policy update across time, but it
    must not silently divide the entire update by the episode length. Invalid
    or degenerate retrieval falls back to the unbiased uniform estimator.
    """
    if length <= 0 or len(weights) != length:
        return [1.0] * max(0, length)
    clean = [max(0.0, float(weight)) for weight in weights]
    total = sum(clean)
    if total <= 1e-8:
        return [1.0] * length
    scale = length / total
    return [weight * scale for weight in clean]


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
    """Observed pre-action boundary pressure used as the deformation baseline.

    The previous implementation inserted BaseBenchmark's exact hunger/fatigue
    drift constants, which was both environment-specific and not a genuine
    counterfactual. The current body state is observable to the organism and
    provides a task-independent baseline for one-step boundary deformation.
    """
    if obs is None or len(obs) < 4:
        return 0.0
    boundary_pressure = float(
        np.max(np.asarray(obs[2:4], dtype=np.float32))
    )
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
    # Lethality is already represented by boundary_target=1.0. Subtracting the
    # pre-action pressure distinguishes sudden collapse from crossing a boundary
    # that was already imminent; forcing every death to 1.0 taught B_psi that
    # arbitrary final actions caused passive starvation.
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


def build_training_param_groups(agent, lr, profile="legacy"):
    """Build optimizer groups with FLC isolated at a lower learning rate.

    Policy-gradient updates on continuous actions are noisy, so the movement
    policy uses a small learning rate. The remaining organism parameters keep
    the prior auxiliary rate, while FLC gets its own much slower group. The FLC
    split prevents future-latent parameters from destabilizing the policy while
    preserving the forward FLC path and trainability.
    """
    if profile not in OPTIMIZER_PROFILES:
        raise ValueError(
            f"unknown optimizer profile={profile!r}; expected one of {OPTIMIZER_PROFILES}"
        )
    multipliers = {
        "legacy": (0.01, 0.1, 0.001),
        "policy_focused": (0.1, 0.01, 0.001),
        "uniform_low": (0.01, 0.01, 0.001),
    }
    policy_multiplier, other_multiplier, flc_multiplier = multipliers[profile]

    flc_param_ids = (
        {id(p) for p in agent.flc.parameters()}
        if hasattr(agent, "flc")
        else set()
    )
    grounded_inverse_param_ids = (
        {
            id(parameter)
            for module in (
                agent.flc.grounded_transfer,
                agent.flc.grounded_motor_projection,
            )
            for parameter in module.parameters()
        }
        if hasattr(agent, "flc")
        and hasattr(agent.flc, "grounded_transfer")
        else set()
    )

    policy_params = []
    flc_params = []
    grounded_inverse_params = []
    other_params = []
    for name, p in agent.named_parameters():
        if not p.requires_grad:
            continue
        if "movement_policy" in name:
            policy_params.append(p)
        elif id(p) in grounded_inverse_param_ids:
            grounded_inverse_params.append(p)
        elif id(p) in flc_param_ids:
            flc_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if policy_params:
        groups.append({"params": policy_params, "lr": lr * policy_multiplier})
    if other_params:
        groups.append({"params": other_params, "lr": lr * other_multiplier})
    if grounded_inverse_params:
        # This path is owned by supervised event-transition reconstruction,
        # not the high-variance policy gradient that forced the generative FLC
        # field onto its much slower learning rate.
        groups.append(
            {"params": grounded_inverse_params, "lr": lr * other_multiplier}
        )
    if flc_params:
        # Evidence from target_threat gradient ablation used lr * 0.001 for
        # FLC. This is 100x lower than the general organism group.
        groups.append({"params": flc_params, "lr": lr * flc_multiplier})
    return groups


def _score_function_policy_loss(
    step_log_probs,
    advantages,
    *,
    blame_weights=None,
    intent_log_prob=None,
    intent_scores=None,
):
    """Compose exact step- and episode-level score-function terms.

    A persistent intent is sampled once before the episode unfolds, so its
    score receives the return-to-go from that first decision exactly once.
    Conditional action proposals remain scored at every step. RTCM weights,
    when explicitly enabled, localize only those per-step proposals; they do
    not alter the causal scope of the episode-level latent variable.
    """
    if len(step_log_probs) != len(advantages):
        raise ValueError("step scores and advantages must have equal length")
    if blame_weights is None:
        blame_weights = [1.0] * len(step_log_probs)
    if len(blame_weights) != len(step_log_probs):
        raise ValueError("blame weights and step scores must have equal length")

    loss = advantages.new_zeros(())
    for log_prob, advantage, causal_weight in zip(
        step_log_probs, advantages, blame_weights, strict=True
    ):
        loss = loss - log_prob.sum() * advantage * causal_weight
    scored_intents = [] if intent_scores is None else list(intent_scores)
    if intent_log_prob is not None:
        scored_intents.append((0, intent_log_prob))
    for start_index, score in scored_intents:
        if not 0 <= start_index < len(advantages):
            raise ValueError("intent score start is outside the episode")
        loss = loss - score.sum() * advantages[start_index]
    return loss


def train_agent(
    agent,
    task_name,
    num_episodes=300,
    lr=0.01,
    seed=None,
    agent_type="agent",
    training_signal="legacy_reward",
    credit_mode="uniform",
    optimizer_profile="legacy",
):
    """
    Train OLF or a baseline from exact rollout policy scores.

    ``credit_mode="uniform"`` is the research default until RTCM attribution
    passes a causal localization test. ``credit_mode="rtcm"`` retains the
    experimental retrograde reweighting as an explicit ablation.
    """
    if training_signal not in TRAINING_SIGNALS:
        raise ValueError(
            f"unknown training_signal={training_signal!r}; expected one of {TRAINING_SIGNALS}"
        )
    if credit_mode not in CREDIT_MODES:
        raise ValueError(
            f"unknown credit_mode={credit_mode!r}; expected one of {CREDIT_MODES}"
        )
    if optimizer_profile not in OPTIMIZER_PROFILES:
        raise ValueError(
            f"unknown optimizer_profile={optimizer_profile!r}; "
            f"expected one of {OPTIMIZER_PROFILES}"
        )
    set_global_seed(seed)
    agent.to(device)
    agent.train()
    agent._training_signal = training_signal
    agent._credit_mode = credit_mode
    # Decoupled optimizers: the policy head needs a much smaller learning
    # rate than the rest of the organism, because policy gradients on
    # continuous actions are high-variance and easily destroy good behavior.
    # The semantics/veto/etc also need a small learning rate because their
    # outputs feed into the policy input — even small changes there shift
    # the policy output. This is constitutional — we still learn everything
    # end-to-end, but at appropriate rates.
    optimizer = optim.Adam(
        build_training_param_groups(agent, lr, profile=optimizer_profile)
    )
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
        episode_policy_log_probs = []
        episode_intent_scores = []
        episode_training_sigmas = []
        episode_training_latents = []
        episode_training_abstract_actions = []
        episode_post_latents = []
        episode_spm_traces = []
        episode_entity_event_masks = []
        episode_observed_effects = []
        episode_homeostatic_deltas = []
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
            policy_log_prob = info_dict.get("_policy_log_prob")
            if policy_log_prob is None:
                raise RuntimeError(
                    "training action did not expose its rollout log-probability"
                )
            episode_policy_log_probs.append(policy_log_prob)
            intent_log_prob = info_dict.get("_intent_log_prob")
            if intent_log_prob is not None:
                episode_intent_scores.append(
                    (len(episode_policy_log_probs) - 1, intent_log_prob)
                )
            if not hasattr(agent, "net"):
                episode_training_sigmas.append(info_dict["_training_sigma"])
                episode_training_latents.append(info_dict["_training_h"])
                episode_spm_traces.append(info_dict["_training_spm_trace"])
                episode_training_abstract_actions.append(
                    info_dict["_training_abstract_action"]
                )
            
            # Step environment
            next_obs, reward, done, info = env.step(action)
            episode_actions.append(action.copy())
            episode_raw_rewards.append(reward)
            
            # Historical behavior is retained only under ``legacy_reward``.
            # The explicit terminal_viability mode below never reads benchmark
            # reward or the semantics value head as a learning signal.
            predicted_value = 0.0
            if training_signal == "legacy_reward" and hasattr(agent, 'semantics'):
                cons = info_dict.get("consequences", None)
                if cons is not None and "value" in cons:
                    predicted_value = cons["value"].mean().item()

            was_lethal = 1.0 if info["status"] in ["death", "starvation"] else 0.0
            learning_signal = _policy_learning_signal(
                reward,
                done=done,
                was_lethal=was_lethal,
                predicted_value=predicted_value,
                training_signal=training_signal,
                self_state=obs[2:4],
                next_self_state=next_obs[2:4],
            )
            consequence_signal = (
                float(reward)
                if training_signal in ("legacy_reward", "raw_reward")
                else learning_signal
            )
            episode_rewards.append(learning_signal)
            episode_dones.append(done)
            episode_infos.append(info)
            episode_self_states.append(obs[2:4].copy())
            
            # Consequence feedback to the fast/slow memory trace
            # Calculate changes in homeostatic self state
            hunger_delta = next_obs[2] - obs[2]
            fatigue_delta = next_obs[3] - obs[3]
            episode_homeostatic_deltas.append(
                (float(hunger_delta), float(fatigue_delta))
            )

            # v0.3.2.11: boundary proximity provides a continuous signal before
            # actual death. The paired zero-action target teaches B_psi the
            # baseline risk of inaction at the same body state.
            episode_boundary_targets.append(
                _boundary_proximity_target(obs, next_obs, was_lethal)
            )
            episode_zero_boundary_targets.append(_passive_boundary_target(obs))
            
            agent.learn_consequence(
                consequence_signal,
                was_lethal,
                hunger_delta,
                fatigue_delta,
                next_obs=next_obs,
            )
            if not hasattr(agent, "net"):
                episode_entity_event_masks.append(
                    agent.last_entity_event_mask.detach().clone()
                )
                episode_observed_effects.append(
                    agent.last_observed_effect.detach().clone()
                )
                episode_post_latents.append(agent.h.detach().clone())
            
            obs = next_obs
            
        # 1. Compute return G_t for policy gradient calculation
        returns = []
        G = 0.0
        return_gamma = (
            1.0
            if training_signal in ("homeostatic_delta", "terminal_homeostasis")
            else 0.95
        )
        for r in reversed(episode_rewards):
            G = r + return_gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(device)

        # The baseline must be independent of the current episode's actions.
        # Use prior episodes only, then append this episode after advantages are
        # fixed. Including the current return biases the score estimator.
        running_mean = (
            sum(_return_history) / len(_return_history)
            if _return_history
            else 0.0
        )
        baseline = torch.tensor(running_mean, device=device)
        advantages = returns - baseline

        ep_total = (
            sum(episode_raw_rewards)
            if training_signal in ("legacy_reward", "raw_reward")
            else sum(episode_rewards)
        )
        _return_history.append(ep_total)
        if len(_return_history) > _baseline_window:
            _return_history.pop(0)

        # 2. Retrieve RTCM retrograde causal blame if a delayed reward or hazard occurs
        # The consequence vector: [final_reward, was_lethal, hunger_delta, fatigue_delta]
        final_info = episode_infos[-1]
        final_was_lethal = 1.0 if final_info["status"] in ["death", "starvation"] else 0.0
        final_reward = sum(episode_rewards)

        consequence_vec = torch.FloatTensor([[final_reward, final_was_lethal, 0.0, 0.0]]).to(device)

        # RTCM credit remains an explicit ablation until its localization is
        # empirically validated. Uniform credit is the unbiased default.
        blame_weights = [1.0] * len(episode_obs)
        if credit_mode == "rtcm" and hasattr(agent, "rtcm"):
            blame_weights = agent.rtcm.retrieve_causal_blame(consequence_vec)
            if not blame_weights:
                blame_weights = [1.0] * len(episode_obs)

        # v3: blend in RTCM delayed-credit blame (top-k softmax retrieval).
        # The delayed-credit weights point at the top-k past steps most likely
        # to have caused the observed consequence. We blend 50/50 with the
        # existing per-step blame so the organism still uses per-step signal
        # but adds delayed causal retrieval. If RTCM returns nothing, the
        # blend leaves the original blame untouched.
        if credit_mode == "rtcm" and hasattr(agent, "rtcm") and not (
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

        blame_weights = _mean_one_credit_weights(
            blame_weights, len(episode_obs)
        )

        # 3. Optimize the exact stochastic proposals recorded during rollout.
        # The previous implementation reset the organism and treated the final
        # motor action as a Normal sample around a second, different latent
        # trajectory. Besides being off-policy, that reset erased RTCM history
        # before its learning step. The score-function estimator belongs to the
        # sampled abstract proposal; boundary and motor transformations are
        # deterministic downstream control.
        optimizer.zero_grad()
        policy_loss = _score_function_policy_loss(
            episode_policy_log_probs,
            advantages,
            blame_weights=(
                blame_weights if not hasattr(agent, "net") else None
            ),
            intent_scores=episode_intent_scores,
        )

        consequence_loss = torch.zeros((), device=device)
        consequence_event_terms = 0
        if not hasattr(agent, "net"):
            if not (
                len(episode_training_sigmas)
                == len(episode_training_latents)
                == len(episode_training_abstract_actions)
                == len(episode_spm_traces)
                == len(episode_entity_event_masks)
                == len(episode_observed_effects)
                == len(episode_post_latents)
                == len(episode_obs)
            ):
                raise RuntimeError("incomplete OLF rollout training trace")

            for idx, sigma_t in enumerate(episode_training_sigmas):
                act_taken = torch.as_tensor(
                    episode_actions[idx], dtype=torch.float32, device=device
                ).unsqueeze(0)

                event_mask = episode_entity_event_masks[idx]
                if bool(event_mask.any()):
                    was_lethal_step = float(
                        episode_infos[idx]["status"] in ("death", "starvation")
                    )
                    consequences_pred = agent.semantics.predict_consequences(
                        sigma_t, act_taken
                    )
                    target_effect = episode_observed_effects[idx].squeeze(0)
                    # Meaning is prospective: an event is valued by the
                    # endogenous viability that unfolds after it, not merely
                    # its immediate body delta. This makes delayed lure and
                    # safe-exit consequences distinguishable without task labels.
                    target_value = returns[idx].detach().reshape(1)
                    target_risk = torch.tensor(
                        [was_lethal_step], device=device
                    )
                    hunger_delta, fatigue_delta = episode_homeostatic_deltas[idx]
                    target_reversibility = torch.tensor(
                        [
                            float(
                                not was_lethal_step
                                and hunger_delta + fatigue_delta < 0.0
                            )
                        ],
                        device=device,
                    )
                    for ent_idx in torch.nonzero(
                        event_mask, as_tuple=False
                    ).flatten().tolist():
                        pred_effect = consequences_pred["dh_pred"][0, ent_idx]
                        pred_value = consequences_pred["value"][0, ent_idx]
                        pred_risk = consequences_pred["terminal_risk"][0, ent_idx]
                        pred_reversibility = consequences_pred["reversibility"][
                            0, ent_idx
                        ]
                        pred_uncertainty = consequences_pred["uncertainty"][
                            0, ent_idx
                        ]
                        effect_loss = nn.functional.mse_loss(
                            pred_effect, target_effect
                        )
                        value_loss = nn.functional.mse_loss(
                            pred_value, target_value
                        )
                        risk_loss = nn.functional.binary_cross_entropy(
                            pred_risk, target_risk
                        )
                        reversibility_loss = nn.functional.binary_cross_entropy(
                            pred_reversibility, target_reversibility
                        )
                        with torch.no_grad():
                            uncertainty_target = torch.clamp(
                                effect_loss.detach()
                                + value_loss.detach()
                                + risk_loss.detach(),
                                0.0,
                                1.0,
                            ).reshape_as(pred_uncertainty)
                        uncertainty_loss = nn.functional.mse_loss(
                            pred_uncertainty, uncertainty_target
                        )
                        consequence_loss = consequence_loss + (
                            effect_loss
                            + value_loss
                            + risk_loss
                            + 0.1 * reversibility_loss
                            + 0.1 * uncertainty_loss
                        )
                        consequence_event_terms += 1

                if bpsi_enabled:
                    with torch.no_grad():
                        consequences_veto = agent.semantics.predict_consequences(
                            sigma_t.detach(), act_taken
                        )
                        dh_pred_veto = consequences_veto["dh_pred"].mean(dim=1)
                        a_zero = torch.zeros_like(act_taken)
                        consequences_zero = agent.semantics.predict_consequences(
                            sigma_t.detach(), a_zero
                        )
                        dh_pred_zero = consequences_zero["dh_pred"].mean(dim=1)

                    was_lethal_step = float(
                        episode_infos[idx]["status"] in ("death", "starvation")
                    )
                    boundary_target = episode_boundary_targets[idx]
                    zero_target = episode_zero_boundary_targets[idx]
                    attributable_risk = _attributable_boundary_target(
                        boundary_target, zero_target, was_lethal_step
                    )
                    action_weight = (
                        12.0
                        if was_lethal_step
                        else (4.0 if attributable_risk > 0.05 else 1.0)
                    )
                    h_t = episode_training_latents[idx]
                    episode_bpsi_data.extend(
                        [
                            (
                                h_t,
                                act_taken.detach(),
                                dh_pred_veto.detach(),
                                torch.tensor(
                                    [[attributable_risk]], device=device
                                ),
                                torch.tensor([[action_weight]], device=device),
                            ),
                            (
                                h_t,
                                a_zero.detach(),
                                dh_pred_zero.detach(),
                                torch.tensor([[0.0]], device=device),
                                torch.tensor([[1.0]], device=device),
                            ),
                        ]
                    )

        prospective_event_losses = []
        prospective_inverse_losses = []
        if not hasattr(agent, "net") and getattr(
            agent, "use_prospective_event_grounding", False
        ):
            max_horizon = agent.prospective_event_field.max_horizon
            for event_idx, event_mask in enumerate(
                episode_entity_event_masks
            ):
                if not bool(event_mask.any()):
                    continue
                start_idx = max(0, event_idx + 1 - max_horizon)
                event_loss = agent.prospective_event_field.event_loss(
                    latents=torch.cat(
                        episode_training_latents[start_idx : event_idx + 1],
                        dim=0,
                    ),
                    sigmas=torch.cat(
                        episode_training_sigmas[start_idx : event_idx + 1],
                        dim=0,
                    ).detach(),
                    actions=torch.cat(
                        episode_training_abstract_actions[
                            start_idx : event_idx + 1
                        ],
                        dim=0,
                    ),
                    effects=torch.cat(
                        episode_observed_effects[start_idx : event_idx + 1],
                        dim=0,
                    ),
                    endpoint=episode_post_latents[event_idx],
                    entity_mask=event_mask,
                    future_value=returns[event_idx].detach(),
                    lethal=float(
                        episode_infos[event_idx]["status"]
                        in ("death", "starvation")
                    ),
                )
                if event_loss is not None:
                    prospective_event_losses.append(event_loss)
                    event_latents = torch.cat(
                        episode_training_latents[
                            start_idx : event_idx + 1
                        ],
                        dim=0,
                    )
                    event_effects = torch.cat(
                        episode_observed_effects[
                            start_idx : event_idx + 1
                        ],
                        dim=0,
                    )
                    endpoint = episode_post_latents[event_idx]
                    inverse_weights = (
                        agent.prospective_event_field.eligibility_weights(
                            event_latents,
                            event_effects,
                            endpoint,
                        ).detach()
                    )
                    event_sigmas = torch.cat(
                        episode_training_sigmas[
                            start_idx : event_idx + 1
                        ],
                        dim=0,
                    ).detach()
                    event_self_state = torch.as_tensor(
                        np.stack(
                            episode_self_states[
                                start_idx : event_idx + 1
                            ]
                        ),
                        dtype=torch.float32,
                        device=device,
                    )
                    prospective_inverse_losses.append(
                        agent.flc.grounded_inverse_loss(
                            current_latents=event_latents,
                            target_future=endpoint.expand_as(event_latents),
                            sigma_flat=event_sigmas.reshape(
                                event_sigmas.shape[0], -1
                            ),
                            self_state=event_self_state,
                            target_actions=torch.cat(
                                episode_training_abstract_actions[
                                    start_idx : event_idx + 1
                                ],
                                dim=0,
                            ),
                            weights=inverse_weights,
                        )
                    )
                    path_length = event_idx + 1 - start_idx
                    sample_count = min(8, path_length)
                    sampled_offsets = torch.linspace(
                        0,
                        path_length - 1,
                        steps=sample_count,
                    ).round().long().unique().tolist()
                    event_entities = torch.nonzero(
                        event_mask, as_tuple=False
                    ).flatten().tolist()
                    event_risk = float(
                        episode_infos[event_idx]["status"]
                        in ("death", "starvation")
                    )
                    for offset in sampled_offsets:
                        trace_idx = start_idx + int(offset)
                        for entity_index in event_entities:
                            agent.prospective_event_memory.add(
                                observation=episode_obs[trace_idx],
                                spm_trace=episode_spm_traces[trace_idx],
                                entity_index=entity_index,
                                endpoint=episode_post_latents[event_idx],
                                future_value=float(returns[event_idx].item()),
                                risk=event_risk,
                                action=episode_training_abstract_actions[
                                    trace_idx
                                ],
                                horizon=event_idx - trace_idx + 1,
                            )

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
                agent,
                episode_obs,
                episode_actions,
                episode_self_states,
                (
                    episode_raw_rewards
                    if training_signal in ("legacy_reward", "raw_reward")
                    else episode_rewards
                ),
            )
            if cf_loss is not None and cf_loss.requires_grad:
                total_loss = (
                    total_loss
                    + FEATURE_FLAGS["counterfactual_loss"] * cf_loss
                )

        # Sparse observation-transition events ground the consequence model.
        # Unlike the old all-zero/coordinate-contact targets, these updates only
        # occur when entity content actually changes.
        # v0.3.2.10: B_psi is NO LONGER trained in the main loss.
        # It gets a dedicated training step below (like RTCM).
        if consequence_event_terms > 0:
            consequence_loss = consequence_loss / consequence_event_terms
            total_loss = total_loss + consequence_loss
        if prospective_event_losses:
            total_loss = total_loss + torch.stack(
                prospective_event_losses
            ).mean()
        if prospective_inverse_losses:
            total_loss = total_loss + torch.stack(
                prospective_inverse_losses
            ).mean()
            
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

        # Slow flow consolidation is intentionally not called here. Its current
        # target is the flow network's own detached output and its input is a
        # synthetic zero observation, making the update circular and unrelated
        # to the transition that occurred. Flow now receives policy gradients
        # through the actual differentiable recoupling path instead.

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

def evaluate_agent(
    agent, task_name, num_episodes=50, seed=None, training_signal=None
):
    """
    Evaluates agent on the specified task and logs metrics.
    """
    set_global_seed(seed)
    agent.eval()
    signal_mode = training_signal or getattr(
        agent, "_training_signal", "legacy_reward"
    )
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
            consequence_signal = (
                float(reward)
                if signal_mode in ("legacy_reward", "raw_reward")
                else _policy_learning_signal(
                    reward,
                    done=done,
                    was_lethal=was_lethal,
                    training_signal=signal_mode,
                    self_state=obs[2:4],
                    next_self_state=next_obs[2:4],
                )
            )
            agent.learn_consequence(
                consequence_signal,
                was_lethal,
                hunger_delta,
                fatigue_delta,
                next_obs=next_obs,
                store=False,
            )
            
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

def _evaluate_with_diagnostics(
    agent, task_name, seed, num_episodes=20, training_signal=None
):
    """Evaluate an agent with diagnostics enabled, returning per-episode metrics."""
    set_global_seed(seed)
    agent.eval()
    agent.diag_mode = True
    signal_mode = training_signal or getattr(
        agent, "_training_signal", "legacy_reward"
    )
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
            consequence_signal = (
                float(reward)
                if signal_mode in ("legacy_reward", "raw_reward")
                else _policy_learning_signal(
                    reward,
                    done=done,
                    was_lethal=was_lethal,
                    training_signal=signal_mode,
                    self_state=obs[2:4],
                    next_self_state=next_obs[2:4],
                )
            )
            agent.learn_consequence(
                consequence_signal,
                was_lethal,
                hunger_delta,
                fatigue_delta,
                next_obs=next_obs,
                store=False,
            )
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

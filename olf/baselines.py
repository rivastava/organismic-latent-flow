"""olf/baselines.py

Standard MLP baseline + ablation harness for the OLF organism.

Constitutional compliance:
- §16 All required ablations and anti-cheat controls.
- §7  Soft risk ablation is implemented as a soft action penalty (per
       user decision), NOT reward shaping.
- §4  'no_memory' disables SPM, RTCM, and ConsequenceMemory.
- §15 'no_rtcm' disables RTCM add_step and blame-weighted training.
"""

import torch
import torch.nn as nn
import numpy as np

from olf.organism import Organism
from olf.motor_memory import _EmptyMotorMemory


class MLPBaselineAgent(nn.Module):
    """Standard MLP reinforcement learning policy.

    Has none of: continuous manifold projection, SPM, RTCM, ConsequenceMemory,
    veto boundary, mode arbitration, or readiness gating.
    """

    def __init__(self, obs_dim=18, action_dim=3, hidden_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def reset_state(self):
        pass

    def learn_consequence(
        self,
        reward,
        was_lethal,
        hunger_delta,
        fatigue_delta,
        next_obs=None,
        store=True,
    ):
        pass

    def select_action(self, obs, evaluate=False):
        device = next(self.parameters()).device
        obs_t = torch.FloatTensor(obs).to(device).unsqueeze(0)
        action_mean = self.net(obs_t)
        policy_log_prob = None
        policy_raw_sample = None
        if not evaluate:
            distribution = torch.distributions.Normal(action_mean, 0.3)
            policy_raw_sample = distribution.sample()
            policy_log_prob = distribution.log_prob(policy_raw_sample).sum(dim=-1)
            action = torch.clamp(policy_raw_sample, -1.0, 1.0)
        else:
            action = action_mean
        return action.squeeze(0).cpu().detach().numpy(), {
            "mode": 0,
            "mode_probs": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "consequences": {"terminal_risk": torch.zeros(1, 2)},
            "impasse": False,
            "risk": 0.0,
            "verdict": "release",
            "_policy_log_prob": policy_log_prob,
            "_policy_mean": action_mean.detach(),
            "_policy_distribution_mean": action_mean.detach(),
            "_policy_raw_sample": (
                None if policy_raw_sample is None else policy_raw_sample.detach()
            ),
            "_policy_exploration_std": 0.0 if evaluate else 0.3,
        }


class _EmptyMemory(nn.Module):
    """Drop-in for ConsequenceMemory that returns no traces."""

    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.register_buffer("trace_active", torch.zeros(1))

    def add_trace(self, *args, **kwargs):
        return None

    def retrieve_trace(self, *args, **kwargs):
        return (
            torch.zeros(1, self.latent_dim),
            torch.zeros(1, 4),
            0.0,
        )


class _ExactMemory(nn.Module):
    """ConsequenceMemory wrapper forcing exact-match episodic retrieval only."""

    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.trace_active = parent.consequence_memory.trace_active

    def retrieve_trace(self, s, a, **kwargs):
        return self.parent.consequence_memory.retrieve_trace(
            s, a, sigma_kernel=0.01
        )


class AblatedOrganism(Organism):
    """OLF Organism with a single constitutional component disabled.

    Supported ablation_type values (matching Constitution §16):
        no_memory
        no_spm
        last_observation_only
        no_rtcm
        no_self_state
        no_abstraction
        exact_episodic_memory_only
        no_veto_boundary
        soft_risk_only                (soft action penalty, not reward shaping)
        no_mode_arbitration
        inspect_only_trace            (forced Inspect mode)
        no_invention
        ungated_invention
        no_future_latent
        no_situation
        no_recoupling_constraint
        no_closure_pressure
        no_diagnostic_decay
    """

    def __init__(
        self,
        obs_dim=18,
        action_dim=3,
        latent_dim=32,
        hidden_dim=64,
        ablation_type=None,
    ):
        super().__init__(obs_dim, action_dim, latent_dim, hidden_dim)
        self.ablation_type = ablation_type

        # If memory is fully off, swap the consequence memory for a no-op stub.
        if ablation_type in ("no_consequence_memory", "no_memory"):
            self.consequence_memory = _EmptyMemory(latent_dim=latent_dim)

        # v3: internal ablation — disable motor_memory (the
        # InventionGenerator's transformation-composition memory).
        if ablation_type == "no_motor_memory":
            self.motor_memory = _EmptyMotorMemory(latent_dim=latent_dim, action_dim=action_dim)

    def select_action(self, obs, evaluate=False):
        device = next(self.parameters()).device
        obs_t = torch.FloatTensor(obs).to(device).unsqueeze(0)

        # 1. Parse observations (supports any num_entities via the new parse_obs).
        if obs_t.shape[-1] == self.obs_dim:
            agent_pos, self_state, context, entities_pos, entities_feats = self.parse_obs(obs_t)
        else:
            pad = torch.zeros(1, self.obs_dim - obs_t.shape[-1], device=device)
            obs_t = torch.cat([obs_t, pad], dim=-1)
            agent_pos, self_state, context, entities_pos, entities_feats = self.parse_obs(obs_t)

        # Ablations: zero selected inputs
        if self.ablation_type == "no_self_state":
            self_state = torch.zeros_like(self_state)
        if self.ablation_type == "no_abstraction":
            entities_feats = torch.zeros_like(entities_feats)
        if self.ablation_type == "last_observation_only":
            self.h = torch.zeros_like(self.h)

        # 2. SPM trace
        if self.ablation_type in ("no_spm", "no_memory"):
            spm_trace = torch.zeros(1, self.latent_dim, device=device)
        else:
            spm_trace = self.spm.get_trace().to(device)

        # 3. Flow update. A consequence observation may already have been
        # integrated by learn_consequence; consume that cached perception once.
        prefetched = self._prefetched_obs
        if prefetched is not None and torch.equal(obs_t, prefetched.to(device)):
            self._prefetched_obs = None
        else:
            self._prefetched_obs = None
            x_flow = torch.cat([obs_t, spm_trace], dim=-1)
            self.h = self.flow(x_flow, self.h.to(device))
        self._h_at_action = self.h.clone().detach()
        self._entity_feats_at_action = entities_feats.detach().clone()

        # 4. Situated binding
        sigma_t = self.semantics.bind(
            spm_trace, entities_pos, entities_feats, context, self_state
        )

        # 5. Movement policy and optional FLC correction form the stochastic
        # abstract-action mean. Sampling happens afterward so the rollout score
        # matches the proposal that is passed to downstream control.
        policy_sigma = sigma_t.detach()
        flat_embeds = policy_sigma.reshape(1, -1)
        policy_inputs = torch.cat([self.h, flat_embeds], dim=-1)
        action_mean = self.movement_policy(policy_inputs)
        if self.ablation_type != "no_future_latent":
            action_mean, _ = self.apply_future_control(
                policy_sigma,
                self_state,
                action_mean,
                context=context,
            )
        (
            a_cand_init,
            policy_log_prob,
            policy_raw_sample,
            exploration_std,
            policy_distribution_mean,
            intent_log_prob,
        ) = self._sample_abstract_action(action_mean, evaluate=evaluate)
        consequences = self.semantics.predict_consequences(sigma_t, a_cand_init)

        impasse_detected = self.impasse_detector.detect()
        uncertainty = consequences["uncertainty"].mean().item()
        risk = consequences["terminal_risk"].max().item()
        closure_pressure = self_state.squeeze(0).cpu().numpy()
        if self.ablation_type == "no_closure_pressure":
            closure_pressure = np.zeros_like(closure_pressure)

        # v0.3.2.8: need_pressure = max(hunger, fatigue) — the dominant unresolved need.
        # Need is not danger. It drives urgency, not rollback.
        need_pressure = float(self_state.squeeze(0).max().item())

        # v0.3.2.6: compute affordance_pressure from consequence predictions
        with torch.no_grad():
            entity_affordance = self._compute_entity_affordance(consequences)
            affordance_pressure = float(entity_affordance.mean().item())

        veto_verdict = self.prev_veto_verdict
        diag_decay = (
            self.diagnostic_tracker.get_decay()
            if hasattr(self, "diagnostic_tracker")
            else 0.0
        )
        readiness = (
            self.readiness_gate.compute_readiness(
                veto_verdict, need_pressure, affordance_pressure=affordance_pressure
            )
            if hasattr(self, "readiness_gate")
            else 1.0
        )

        # 6. Mode arbitration
        if self.ablation_type == "no_mode_arbitration":
            mode = torch.tensor([0], device=device)
            mode_probs = torch.zeros(1, 8, device=device)
            mode_probs[0, 0] = 1.0
        elif self.ablation_type == "inspect_only_trace":
            mode = torch.tensor([1], device=device)
            mode_probs = torch.zeros(1, 8, device=device)
            mode_probs[0, 1] = 1.0
        else:
            mode, mode_probs = self.arbitrator(
                self.h,
                uncertainty,
                impasse_detected,
                risk,
                closure_pressure,
                self.recent_consequence_val,
                diagnostic_decay=diag_decay,
                readiness=readiness,
                veto_verdict=veto_verdict,
                recoupling_required=self._recouple_required if hasattr(self, '_recouple_required') else False
            )

        # 7. Invention
        if self.ablation_type == "no_invention":
            a_cand = a_cand_init
        elif self.ablation_type == "ungated_invention" or mode[0] == 2:
            sig = sigma_t
            if self.ablation_type == "no_situation":
                sig = torch.zeros_like(sigma_t)
            mm = getattr(self, "motor_memory", None)
            if self.ablation_type in ("no_consequence_memory", "no_memory"):
                a_cand = self.invention(self.h, sig, self.consequence_memory, motor_memory=mm)
            elif self.ablation_type == "exact_episodic_memory_only":
                a_cand = self.invention(self.h, sig, _ExactMemory(self), motor_memory=mm)
            else:
                a_cand = self.invention(self.h, sig, self.consequence_memory, motor_memory=mm)
        else:
            a_cand = a_cand_init

        consequences_final = self.semantics.predict_consequences(sigma_t, a_cand)

        # 8. Veto boundary
        danger = 0.0
        if self.ablation_type == "no_veto_boundary":
            a_steered = a_cand
            veto_verdict = "release"
            viability = 1.0
        elif self.ablation_type == "soft_risk_only":
            # Soft action penalty: scale action by (1 - risk) instead of hard-gating.
            # Per user decision: this is a behavioral soft risk, NOT reward shaping.
            soft_scale = float(max(0.05, 1.0 - risk))
            a_steered = a_cand * soft_scale
            veto_verdict = "release"
            viability = soft_scale
        else:
            # v0.3.2.10 — Boundary Deformation Risk.
            # The Veto uses its own B_psi(h, a, dh_pred) for boundary risk.
            # No FiLM-based danger computation needed.
            a_steered, veto_verdict, viability, _, danger, _ = self.veto.constrain_release(
                self.h, a_cand, sigma_t, self.semantics
            )

        self.prev_veto_verdict = veto_verdict

        # 9. Motor release (v0.3.2.2: readiness_factor passed through)
        act_np = a_steered.squeeze(0).cpu().detach().numpy()
        # Ablated organisms don't have readiness gate — use 1.0 (no modulation)
        abl_readiness = 1.0
        if (
            self.ablation_type == "no_recoupling_constraint"
            and veto_verdict == "recouple"
        ):
            a_final = self.motor.process_release(act_np, "release", mode.item(), viability, abl_readiness)
        else:
            a_final = self.motor.process_release(act_np, veto_verdict, mode.item(), viability, abl_readiness)

        # 10. State trace updates (respecting ablations)
        if self.ablation_type != "no_diagnostic_decay":
            if self.ablation_type not in ("no_spm", "no_memory"):
                self.spm.update(self.h)
        if self.ablation_type not in ("no_rtcm", "no_memory"):
            self.rtcm.add_step(self.h, a_final, h_next=None)
        self.impasse_detector.add_step(agent_pos.squeeze(0).cpu().numpy(), a_final)

        self.last_sigma = sigma_t.clone().detach()
        self.last_action = a_final.copy()

        return a_final, {
            "mode": mode.item(),
            "mode_probs": mode_probs.squeeze(0).cpu().detach().numpy(),
            "consequences": consequences_final,
            "impasse": impasse_detected,
            "risk": risk,
            "danger": danger,
            "verdict": veto_verdict,
            "_policy_log_prob": policy_log_prob,
            "_policy_mean": action_mean.detach(),
            "_policy_distribution_mean": policy_distribution_mean,
            "_policy_raw_sample": policy_raw_sample,
            "_policy_exploration_std": exploration_std,
            "_policy_exploration_intent": self._exploration_intent.detach(),
            "_intent_log_prob": intent_log_prob,
            "_episode_intent": (
                None
                if self._episode_intent is None
                else self._episode_intent.detach()
            ),
            "_episode_intent_raw_sample": self._episode_intent_raw_sample,
            "_episode_intent_distribution_mean": (
                self._episode_intent_distribution_mean
            ),
            "_episode_intent_source": self._episode_intent_source,
            "_training_h": self.h.detach(),
            "_training_sigma": sigma_t,
            "_training_spm_trace": spm_trace.detach(),
            "_training_abstract_action": a_cand_init.detach(),
        }

    def learn_consequence(
        self,
        reward,
        was_lethal,
        hunger_delta,
        fatigue_delta,
        next_obs=None,
        store=True,
    ):
        # 'no_consequence_memory' / 'no_memory' disables fast-trace storage.
        if self.ablation_type in ("no_consequence_memory", "no_memory"):
            s_t = self._h_at_action if self._h_at_action is not None else self.h
            s_after = self._recouple_observation(
                next_obs,
                s_t=s_t,
                track_grad=bool(store and self.training),
            )
            consequence_vec = np.array(
                [reward, float(was_lethal), hunger_delta, fatigue_delta],
                dtype=np.float32,
            )
            if self.ablation_type != "no_memory":
                self.rtcm.complete_last_step(consequence_vec, s_after)
            self.recent_consequence_val = (
                -float(hunger_delta)
                - float(fatigue_delta)
                - float(was_lethal)
            )
            self.readiness_gate.update(self.h, received_consequence=True)
            return

        # 'soft_risk_only' now uses soft action penalty, not reward shaping.
        # We do NOT modify reward — the penalty was applied in select_action.
        super().learn_consequence(
            reward,
            was_lethal,
            hunger_delta,
            fatigue_delta,
            next_obs=next_obs,
            store=store,
        )

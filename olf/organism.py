import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import replace

from olf.flow import LTCFlow
from olf.spm import SphericalPhaseMemory
from olf.rtcm import RetrogradeTemporalCausalMemory
from olf.consequence_memory import ConsequenceMemory
from olf.semantics import TriadicSemantics
from olf.boundary import VetoBoundary
from olf.future import FutureLatentControl
from olf.arbitrator import ModeArbitrator, ImpasseDetector, ReadinessGate, DiagnosticDecayTracker
from olf.invention import InventionGenerator
from olf.motor import MotorCortex
from olf.motor_memory import MotorMemory
from olf.attractor import AttractorField
from olf.salience import ProspectiveSalienceGate
from olf.events import entity_feature_event_mask
from olf.geometry import exponential_map, log_map_sphere, project_to_tangent
from olf.prospective import ProspectiveEventField
from olf.prospective_memory import ProspectiveEventMemory
from olf.ghosts.config import GhostConfig
from olf.ghosts.integration import GhostIntegration

class Organism(nn.Module):
    """
    Organism integrates all OLF Core systems:
    Manifold Flow, SPM, RTCM, Consequence Memory, Situated Semantics, Veto,
    Arbitrator, Readiness Gate, Diagnostic Decay, and Motor.
    
    Constitution compliance:
    - §1: Single continuous evolving state h(t)
    - §2: h(t) ∈ S^{d-1} via LTC + sphere projection
    - §3: Meaning through consequence prediction
    - §7: Hard veto boundary, not reward shaping
    - §9: Recoupling enforced via motor cortex
    - §10: Invention gated by impasse
    - §11: Soft mode biases, not hard symbolic overrides
    - §12: Readiness gate separates action pressure from release
    - §13: Diagnostic decay prevents infinite inspection
    - §14: Closure pressure from unresolved contradiction
    """
    def __init__(
        self,
        obs_dim=18,
        action_dim=3,
        latent_dim=32,
        hidden_dim=64,
        randomize_initial_latent=False,
        exploration_correlation=0.0,
        exploration_intent_scale=0.0,
        use_consequence_future_hint=True,
        use_hierarchical_intent=False,
        hierarchical_intent_std=1.0,
        hierarchical_intent_blend=0.8,
        hierarchical_babble_probability=0.0,
        use_prospective_event_grounding=False,
        prospective_max_horizon=64,
        use_prospective_event_memory=True,
        use_situated_prospective_keys=True,
        use_prospective_action_retrieval=True,
        ghost_mode: str = "off",
        ghost_config: GhostConfig | None = None,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.randomize_initial_latent = bool(randomize_initial_latent)
        if not 0.0 <= exploration_correlation < 1.0:
            raise ValueError("exploration_correlation must be in [0, 1)")
        self.exploration_correlation = float(exploration_correlation)
        if exploration_intent_scale < 0.0:
            raise ValueError("exploration_intent_scale must be non-negative")
        self.exploration_intent_scale = float(exploration_intent_scale)
        self.use_consequence_future_hint = bool(use_consequence_future_hint)
        self.use_hierarchical_intent = bool(use_hierarchical_intent)
        if hierarchical_intent_std <= 0.0:
            raise ValueError("hierarchical_intent_std must be positive")
        if not 0.0 <= hierarchical_intent_blend <= 1.0:
            raise ValueError("hierarchical_intent_blend must be in [0, 1]")
        self.hierarchical_intent_std = float(hierarchical_intent_std)
        self.hierarchical_intent_blend = float(hierarchical_intent_blend)
        if not 0.0 <= hierarchical_babble_probability <= 1.0:
            raise ValueError(
                "hierarchical_babble_probability must be in [0, 1]"
            )
        self.hierarchical_babble_probability = float(
            hierarchical_babble_probability
        )
        self.use_prospective_event_grounding = bool(
            use_prospective_event_grounding
        )
        if prospective_max_horizon < 1:
            raise ValueError("prospective_max_horizon must be positive")
        self.prospective_max_horizon = int(prospective_max_horizon)
        self.use_prospective_event_memory = bool(
            use_prospective_event_memory
        )
        self.use_situated_prospective_keys = bool(
            use_situated_prospective_keys
        )
        self.use_prospective_action_retrieval = bool(
            use_prospective_action_retrieval
        )
        self.explore_noise_init = 0.3  # Initial exploration noise std
        self.explore_noise_min = 0.05  # Minimum exploration noise

        # Infer num_entities from observation layout: obs = agent(2) + self_state(2) + context(2) + N * (rel_pos(2) + feats(4))
        remaining = obs_dim - 6
        if remaining <= 0 or remaining % 6 != 0:
            raise ValueError(
                f"obs_dim={obs_dim} must equal 6 + 6 * num_entities (got residual {remaining})"
            )
        self.num_entities = remaining // 6

        # Submodules
        self.flow = LTCFlow(input_dim=obs_dim + latent_dim, hidden_dim=latent_dim)
        self.spm = SphericalPhaseMemory(latent_dim=latent_dim)
        self.rtcm = RetrogradeTemporalCausalMemory(latent_dim=latent_dim, action_dim=action_dim)
        self.consequence_memory = ConsequenceMemory(trace_dim=hidden_dim, action_dim=action_dim, latent_dim=latent_dim)

        # motor transformation memory. Stores (before_h, action, after_h,
        # success, reward) with explicit delta. InventionGenerator queries
        # this to compose candidate action sequences from previously-observed
        # successful transformations, rather than just replaying old actions.
        self.motor_memory = MotorMemory(latent_dim=latent_dim, action_dim=action_dim)
        self.semantics = TriadicSemantics(
            spm_dim=latent_dim,
            entity_dim=6,
            context_dim=2,
            self_state_dim=2,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        self.veto = VetoBoundary(latent_dim=latent_dim, action_dim=action_dim)
        self.flc = FutureLatentControl(
            latent_dim=latent_dim,
            action_dim=action_dim,
            sigma_dim=hidden_dim * self.num_entities,
            self_state_dim=2,
            hidden_dim=hidden_dim,
        )
        self.arbitrator = ModeArbitrator(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.impasse_detector = ImpasseDetector()
        self.invention = InventionGenerator(latent_dim=latent_dim, action_dim=action_dim, hidden_dim=hidden_dim)
        self.motor = MotorCortex()

        # Readiness gate
        self.readiness_gate = ReadinessGate()

        # Diagnostic decay tracker
        self.diagnostic_tracker = DiagnosticDecayTracker()

        # Attractor field. Goals are attractors in
        # latent space, not symbolic commands.
        self.attractor_field = AttractorField(latent_dim=latent_dim)

        # RTCM design: Prospective salience gate. Decides
        # what events to write into long-term memory based on estimated
        # future causal value, not just because they happened.
        self.salience_gate = ProspectiveSalienceGate(latent_dim=latent_dim)
        # An optional research module must not perturb the initialization of
        # established downstream policy heads merely by existing. Forking the
        # RNG keeps its own random initialization deterministic while restoring
        # the organism's construction stream afterward.
        with torch.random.fork_rng(devices=[]):
            self.prospective_event_field = ProspectiveEventField(
                latent_dim=latent_dim,
                sigma_dim=hidden_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                max_horizon=self.prospective_max_horizon,
            )
        self.prospective_event_memory = ProspectiveEventMemory(
            obs_dim=obs_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

        # Policy model for physical moves (dx, dy, u)
        # policy emerges from the latent flow, but for the
        # prototype uses a small MLP head that maps the bound sigma
        # plus the latent state to a 3-dim action (dx, dy, u). The `u` action
        # is treated as continuous in [-1, 1] for likelihood computation but
        # the env can threshold it (e.g. > 0.5 = use).
        self.movement_policy = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim * self.num_entities, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh()
        )

        # Xavier initialization for movement policy
        for layer in self.movement_policy:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        # Warmup counter: FLC steering is disabled for first N episodes
        self.warmup_episodes = 30
        self.episode_count = 0

        # stateful recoupling obligation. When a world-mutating
        # action is released, recouple_required becomes True. The next
        # real observation/consequence satisfies the obligation, and
        # readiness is restored. Do NOT fake perception inside select_action.
        self._recouple_required = False

        # Last-mode memory: carries across select_action calls until motor release
        self.prev_veto_verdict = "release"

        # Snapshot of h at the start of select_action (for honest s_{t+1} in trace)
        self._h_at_action = None
        self._prefetched_obs = None

        # diagnostic instrumentation. When diag_mode is True, the
        # organism records per-step telemetry into diag_buffer for the
        # diagnostic report. Forward pass, gradient flow, and any
        # trainable parameters are unchanged. Default is off.
        self.diag_mode: bool = False
        self.diag_buffer: list = []
        self.diag_episode: int = 0

        # Persistent memories and learned transfer fields require one shared
        # manifold coordinate frame. Randomizing h_0 on every episode makes the
        # same situation start from an unrelated sphere direction. Sample one
        # per-organism origin; keep random resets only as an explicit ablation.
        initial_latent = torch.randn(1, latent_dim)
        initial_latent = F.normalize(initial_latent, p=2, dim=-1)
        self.register_buffer("initial_latent", initial_latent)
        self.register_buffer(
            "consequence_events_seen", torch.zeros((), dtype=torch.long)
        )
        self.register_buffer(
            "prospective_events_seen", torch.zeros((), dtype=torch.long)
        )

        # role-free ghost population integration. The organism OWNS the
        # ghost subsystem; it is constructed with a reference to this Organism
        # so every cognitive operation (FLC, transfer, boundary, motor,
        # semantics) is the organism's own instance. When ghost_mode == "off"
        # no subsystem is created and behavior is byte-identical to before.
        self.ghost_mode = ghost_mode
        if ghost_config is None:
            cfg = GhostConfig(
                ghost_mode=ghost_mode,
                latent_dim=latent_dim,
                action_dim=action_dim,
            )
        else:
            cfg = replace(
                ghost_config,
                ghost_mode=ghost_mode,
                latent_dim=latent_dim,
                action_dim=action_dim,
            )
        self.ghost_config = cfg
        self.ghost: GhostIntegration | None = None
        if cfg.active:
            self.ghost = GhostIntegration(cfg, organism=self)
        self._ghost_token = None
        self._ghost_base_future = None

        self.reset_state()
        
    def reset_diag(self):
        """reset the diagnostic buffer. Called by the diagnostic
        runner between episodes. Telemetry only — no behavior change.
        """
        self.diag_buffer = []
        self.diag_episode = 0
        # Reset per-call counters on child modules
        if hasattr(self, "motor_memory"):
            self.motor_memory._query_count = 0

    def self_state_swap_probe(self, obs):
        """measure whether self_state changes downstream behavior.

        Binds the same observation twice — once with the original self_state
        and once with the flipped self_state (1.0 - ss, clamped to [0,1]).
        Returns a dict of L2 differences for:
          - sigma (the bound representation)
          - consequence value predictions
          - consequence risk predictions
          - readiness scores
          - mode logits (before argmax)
          - action proposals (before veto)
        All under torch.no_grad() — pure diagnostic, no gradient.
        """
        device = next(self.parameters()).device
        obs_t = torch.FloatTensor(obs).to(device).unsqueeze(0)

        agent_pos, self_state, context, entities_pos, entities_feats = self.parse_obs(obs_t)
        spm_trace = self.spm.get_trace().to(device)

        # --- Original ---
        sigma_orig = self.semantics.bind(spm_trace, entities_pos, entities_feats, context, self_state)
        flat_orig = sigma_orig.reshape(1, -1)
        policy_orig = torch.cat([self.h, flat_orig], dim=-1)
        a_orig = self.movement_policy(policy_orig)
        cons_orig = self.semantics.predict_consequences(sigma_orig, a_orig)

        # --- Flipped self_state ---
        self_state_flip = torch.clamp(1.0 - self_state, 0.0, 1.0)
        sigma_flip = self.semantics.bind(spm_trace, entities_pos, entities_feats, context, self_state_flip)
        flat_flip = sigma_flip.reshape(1, -1)
        policy_flip = torch.cat([self.h, flat_flip], dim=-1)
        a_flip = self.movement_policy(policy_flip)
        cons_flip = self.semantics.predict_consequences(sigma_flip, a_flip)

        with torch.no_grad():
            sigma_l2 = float((sigma_orig - sigma_flip).reshape(-1).norm().item())
            value_diff = float((cons_orig["value"] - cons_flip["value"]).abs().mean().item())
            risk_diff = float((cons_orig["terminal_risk"] - cons_flip["terminal_risk"]).abs().mean().item())
            uncertainty_diff = float((cons_orig["uncertainty"] - cons_flip["uncertainty"]).abs().mean().item())
            action_diff = float((a_orig - a_flip).reshape(-1).norm().item())

            # compute affordance_pressure for both self_states
            af_orig = float((cons_orig["value"].squeeze(-1) - cons_orig["terminal_risk"].squeeze(-1) - cons_orig["uncertainty"].squeeze(-1)).mean().item())
            af_flip = float((cons_flip["value"].squeeze(-1) - cons_flip["terminal_risk"].squeeze(-1) - cons_flip["uncertainty"].squeeze(-1)).mean().item())
            affordance_pressure_diff = float(abs(af_orig - af_flip))

            # Mode difference: run arbitrator with both consequence sets
            impasse = self.impasse_detector.detect()
            uncertainty_o = cons_orig["uncertainty"].mean().item()
            risk_o = cons_orig["terminal_risk"].max().item()
            closure_p = self_state.squeeze(0).cpu().numpy()
            diag_decay = self.diagnostic_tracker.get_decay()
            # need_pressure for readiness (not risk)
            need_o = float(self_state.squeeze(0).max().item())
            readiness_o = self.readiness_gate.compute_readiness(
                self.prev_veto_verdict, need_o, affordance_pressure=af_orig
            )

            mode_o, _ = self.arbitrator(
                self.h, uncertainty_o, impasse, risk_o, closure_p,
                self.recent_consequence_val,
                diagnostic_decay=diag_decay, readiness=readiness_o,
                veto_verdict=self.prev_veto_verdict,
                recoupling_required=self._recouple_required,
            )

            uncertainty_f = cons_flip["uncertainty"].mean().item()
            risk_f = cons_flip["terminal_risk"].max().item()
            # need_pressure for readiness (not risk)
            need_f = float(self_state_flip.squeeze(0).max().item())
            readiness_f = self.readiness_gate.compute_readiness(
                self.prev_veto_verdict, need_f, affordance_pressure=af_flip
            )

            mode_f, _ = self.arbitrator(
                self.h, uncertainty_f, impasse, risk_f, closure_p,
                self.recent_consequence_val,
                diagnostic_decay=diag_decay, readiness=readiness_f,
                veto_verdict=self.prev_veto_verdict,
                recoupling_required=self._recouple_required,
            )

            mode_diff = int(mode_o.item() != mode_f.item())
            readiness_diff = float(abs(readiness_o - readiness_f))

        return {
            "sigma_l2": sigma_l2,
            "value_diff": value_diff,
            "risk_diff": risk_diff,
            "uncertainty_diff": uncertainty_diff,
            "affordance_pressure_diff": affordance_pressure_diff,
            "action_diff": action_diff,
            "readiness_diff": readiness_diff,
            "mode_diff": mode_diff,
        }

    def reset_state(self):
        """Resets agent states, memory history, and motor controls."""
        if self.randomize_initial_latent:
            h_init = torch.randn(
                1, self.latent_dim, device=self.initial_latent.device
            )
            self.h = F.normalize(h_init, p=2, dim=-1)
        else:
            self.h = self.initial_latent.clone()
        self.spm.reset_memory()
        self.rtcm.reset_history()
        self.impasse_detector.reset()
        self.motor.reset()
        self.readiness_gate.reset()
        self.diagnostic_tracker.reset()

        # Update warmup state based on episode count.
        # If the caller already set episode_count for this episode, keep it.
        # Otherwise, auto-increment from the previous value (covers run loops
        # that simply call reset_state() each episode).
        if not getattr(self, "_episode_count_locked", False):
            self.episode_count = getattr(self, "episode_count", 0) + 1
        self.veto.warmup = (self.episode_count < self.warmup_episodes)

        # Reset per-step caches
        self.last_sigma = None
        self.last_action = None
        self._h_at_action = None
        self._prefetched_obs = None
        self._entity_feats_at_action = None
        self.last_entity_event_mask = torch.zeros(
            self.num_entities, dtype=torch.bool, device=self.h.device
        )
        self.last_observed_effect = torch.zeros(
            1, self.latent_dim, device=self.h.device
        )
        self._exploration_state = torch.zeros(
            1, self.action_dim, device=self.h.device
        )
        self._exploration_initialized = False
        self._episode_intent = None
        self._episode_intent_raw_sample = None
        self._episode_intent_distribution_mean = None
        self._episode_intent_source = None
        if self.training and self.exploration_intent_scale > 0.0:
            intent = torch.randn(
                1, self.action_dim, device=self.h.device
            )
            self._exploration_intent = F.normalize(
                intent, p=2, dim=-1
            ) * self.exploration_intent_scale
        else:
            self._exploration_intent = torch.zeros_like(
                self._exploration_state
            )
        self.recent_consequence_val = 0.0

        # The veto starts the episode with a clean slate
        self.prev_veto_verdict = "release"
        self._recouple_required = False

        # clear ghost pending-release token and reset the (temporary)
        # ghost population. The learned reachable-deformation space persists.
        # An episode reset explicitly aborts any pending release transaction and
        # records that it did so.
        self._ghost_reset_aborted_pending = self._ghost_token is not None
        self._ghost_token = None
        self._ghost_base_future = None
        if self.ghost is not None:
            self.ghost.reset()
        
    def parse_obs(self, obs_tensor):
        """Parses observation vector for arbitrary num_entities.

        Layout: [agent(2), self_state(2), context(2), e1_pos(2), e1_feats(4), ..., eN_pos(2), eN_feats(4)]
        Returns: agent_pos, self_state, context, entities_pos(B, N, 2), entities_feats(B, N, 4)
        """
        agent_pos = obs_tensor[:, 0:2]
        self_state = obs_tensor[:, 2:4]
        context = obs_tensor[:, 4:6]

        entities_pos = []
        entities_feats = []
        for i in range(self.num_entities):
            base = 6 + i * 6
            entities_pos.append(obs_tensor[:, base:base + 2])
            entities_feats.append(obs_tensor[:, base + 2:base + 6])

        if entities_pos:
            entities_pos = torch.stack(entities_pos, dim=1)
            entities_feats = torch.stack(entities_feats, dim=1)
        else:
            entities_pos = torch.zeros(1, 0, 2, device=obs_tensor.device)
            entities_feats = torch.zeros(1, 0, 4, device=obs_tensor.device)

        return agent_pos, self_state, context, entities_pos, entities_feats

    def apply_future_control(
        self,
        sigma_t,
        self_state,
        base_action,
        context=None,
    ):
        """Apply FLC using a consequence-grounded future when available."""
        sigma_flat = sigma_t.reshape(base_action.shape[0], -1)
        confidence = self._future_grounding_confidence()
        future_hint = None
        grounded_action_hint = None
        action_hint_confidence = 0.0
        if self.use_consequence_future_hint and confidence > 0.0:
            with torch.no_grad():
                if self.use_prospective_event_grounding:
                    remembered = (
                        self._read_prospective_memory(
                            sigma_t,
                            query_self_state=self_state,
                            query_context=context,
                        )
                        if self.use_prospective_event_memory
                        else None
                    )
                    if remembered is not None:
                        prospective_value = (
                            remembered["value"].squeeze(-1)
                            - remembered["risk"].squeeze(-1)
                        )
                        support = remembered["support"]
                        weights = torch.softmax(
                            prospective_value
                            + torch.log(support.clamp_min(1e-8)),
                            dim=-1,
                        )
                        future_hint = F.normalize(
                            (
                                weights.unsqueeze(-1)
                                * remembered["future_latent"]
                            ).sum(dim=1),
                            p=2,
                            dim=-1,
                        )
                        if self.use_prospective_action_retrieval:
                            grounded_action_hint = (
                                weights.unsqueeze(-1)
                                * remembered["action"]
                            ).sum(dim=1)
                        support_confidence = float(
                            (weights * support).sum(dim=-1).mean().item()
                        )
                        confidence = min(confidence, support_confidence)
                        if self.use_prospective_action_retrieval:
                            action_hint_confidence = confidence
                    else:
                        prospective = self.prospective_event_field(
                            self.h, sigma_t, base_action
                        )
                        prospective_value = (
                            prospective["value"].squeeze(-1)
                            - prospective["risk"].squeeze(-1)
                        )
                        weights = torch.softmax(
                            prospective_value, dim=-1
                        )
                        future_hint = F.normalize(
                            (
                                weights.unsqueeze(-1)
                                * prospective["future_latent"]
                            ).sum(dim=1),
                            p=2,
                            dim=-1,
                        )
                else:
                    consequences = self.semantics.predict_consequences(
                        sigma_t, base_action
                    )
                    prospective_value = self._compute_entity_affordance(
                        consequences
                    )
                    weights = torch.softmax(prospective_value, dim=-1)
                    predicted_effect = (
                        weights.unsqueeze(-1) * consequences["dh_pred"]
                    ).sum(dim=1)
                    predicted_effect = project_to_tangent(
                        self.h, predicted_effect
                    )
                    future_hint = exponential_map(self.h, predicted_effect)
        return self.flc(
            self.h,
            sigma_flat,
            self_state,
            base_action,
            future_hint=future_hint,
            hint_confidence=(
                confidence if self.use_consequence_future_hint else 0.0
            ),
            use_grounded_inverse=self.use_prospective_event_grounding,
            grounded_action_hint=grounded_action_hint,
            action_hint_confidence=action_hint_confidence,
        )

    @staticmethod
    def _situated_memory_key(sigma, self_state, context):
        entity_count = sigma.shape[1]
        body = F.normalize(self_state, p=2, dim=-1).unsqueeze(1).expand(
            -1, entity_count, -1
        )
        situation = F.normalize(context, p=2, dim=-1).unsqueeze(1).expand(
            -1, entity_count, -1
        )
        return F.normalize(
            torch.cat(
                [F.normalize(sigma, p=2, dim=-1), body, situation],
                dim=-1,
            ),
            p=2,
            dim=-1,
        )

    def _read_prospective_memory(
        self,
        query_sigma,
        query_self_state=None,
        query_context=None,
    ):
        return self._read_prospective_store(
            self.prospective_event_memory,
            query_sigma,
            query_self_state=query_self_state,
            query_context=query_context,
        )

    def _read_prospective_store(
        self,
        memory,
        query_sigma,
        *,
        query_self_state,
        query_context,
    ):
        records = memory.records()
        if records is None:
            return None
        _, memory_self_state, memory_context, memory_pos, memory_feats = (
            self.parse_obs(records["observations"])
        )
        memory_sigmas = self.semantics.bind(
            records["spm_traces"],
            memory_pos,
            memory_feats,
            memory_context,
            memory_self_state,
        )
        rows = torch.arange(
            memory_sigmas.shape[0], device=memory_sigmas.device
        )
        memory_situated_keys = self._situated_memory_key(
            memory_sigmas, memory_self_state, memory_context
        )
        memory_keys = memory_situated_keys[
            rows, records["entity_indices"]
        ]
        if (
            not self.use_situated_prospective_keys
            or query_self_state is None
            or query_context is None
        ):
            query_keys = query_sigma
            memory_keys = memory_sigmas[rows, records["entity_indices"]]
        else:
            query_keys = self._situated_memory_key(
                query_sigma, query_self_state, query_context
            )
        return memory.read(query_keys, memory_keys)

    def _sample_abstract_action(self, action_mean, evaluate=False):
        """Sample the stochastic action proposal whose score is optimized.

        Boundary refinement and motor release remain deterministic downstream
        transformations.  The policy-gradient score therefore belongs to this
        abstract proposal, not to the transformed motor action returned to the
        environment.
        """
        intent_log_prob = None
        hierarchical_mean = action_mean
        if self.use_hierarchical_intent:
            if self._episode_intent is None:
                if evaluate:
                    self._episode_intent = torch.clamp(
                        action_mean.detach(), -1.0, 1.0
                    )
                else:
                    babble = (
                        self.hierarchical_babble_probability > 0.0
                        and bool(
                            torch.rand((), device=action_mean.device)
                            < self.hierarchical_babble_probability
                        )
                    )
                    if babble:
                        raw_intent = torch.empty_like(action_mean).uniform_(
                            -1.0, 1.0
                        )
                        self._episode_intent_source = "babble"
                    else:
                        intent_distribution = torch.distributions.Normal(
                            action_mean, self.hierarchical_intent_std
                        )
                        raw_intent = intent_distribution.sample()
                        intent_log_prob = intent_distribution.log_prob(
                            raw_intent
                        ).sum(dim=-1)
                        self._episode_intent_source = "policy"
                    self._episode_intent_raw_sample = raw_intent.detach()
                    self._episode_intent_distribution_mean = (
                        action_mean.detach()
                    )
                    self._episode_intent = torch.clamp(
                        raw_intent, -1.0, 1.0
                    ).detach()
            blend = self.hierarchical_intent_blend
            hierarchical_mean = (
                (1.0 - blend) * action_mean
                + blend * self._episode_intent
            )

        if evaluate:
            return (
                torch.clamp(hierarchical_mean, -1.0, 1.0),
                None,
                None,
                0.0,
                hierarchical_mean.detach(),
                intent_log_prob,
            )

        decay = max(
            0.0, 1.0 - self.episode_count / (self.warmup_episodes * 5)
        )
        noise_std = self.explore_noise_min + (
            self.explore_noise_init - self.explore_noise_min
        ) * decay
        exploratory_mean = hierarchical_mean + self._exploration_intent
        rho = self.exploration_correlation
        if rho > 0.0 and self._exploration_initialized:
            distribution_mean = exploratory_mean + rho * self._exploration_state
            innovation_std = noise_std * (1.0 - rho * rho) ** 0.5
        else:
            distribution_mean = exploratory_mean
            innovation_std = noise_std
        distribution = torch.distributions.Normal(
            distribution_mean, innovation_std
        )
        raw_sample = distribution.sample()
        log_prob = distribution.log_prob(raw_sample).sum(dim=-1)
        sampled_action = torch.clamp(raw_sample, -1.0, 1.0)
        self._exploration_state = (raw_sample - exploratory_mean).detach()
        self._exploration_initialized = True
        return (
            sampled_action,
            log_prob,
            raw_sample.detach(),
            float(innovation_std),
            distribution_mean.detach(),
            intent_log_prob,
        )

    def renew_episode_intent(self):
        """End the current temporal intention at an observed event boundary."""
        self._episode_intent = None
        self._episode_intent_raw_sample = None
        self._episode_intent_distribution_mean = None
        self._episode_intent_source = None

    def _recouple_observation(self, next_obs, s_t=None, track_grad=True):
        """Integrate the post-action observation exactly once.

        Storage ablations may remove memories, but they must preserve this
        perception timing or the ablation would change the organismic loop in
        addition to the named mechanism.
        """
        if next_obs is None:
            return self.h.clone().detach()
        device = next(self.parameters()).device
        next_obs_t = torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        ).reshape(1, -1)
        if next_obs_t.shape[-1] < self.obs_dim:
            pad = torch.zeros(1, self.obs_dim - next_obs_t.shape[-1], device=device)
            next_obs_t = torch.cat([next_obs_t, pad], dim=-1)
        elif next_obs_t.shape[-1] > self.obs_dim:
            next_obs_t = next_obs_t[..., : self.obs_dim]

        if getattr(self, "ablation_type", None) in ("no_spm", "no_memory"):
            spm_trace = torch.zeros(1, self.latent_dim, device=device)
        else:
            spm_trace = self.spm.get_trace().to(device)
        latent_before = self.h if s_t is None else s_t
        x_flow = torch.cat([next_obs_t, spm_trace], dim=-1)
        if track_grad:
            s_after = self.flow(x_flow, latent_before.to(device))
        else:
            with torch.no_grad():
                s_after = self.flow(x_flow, latent_before.to(device)).detach()
        self.h = s_after
        self._prefetched_obs = next_obs_t.detach().clone()
        return s_after

    @staticmethod
    def _compute_entity_affordance(consequences):
        """Extract per-entity affordance from consequence predictions.

        affordance_i = value_i - risk_i - uncertainty_i.

        Used by both action steering (directional gradient) and readiness
        gating (scalar pressure). Extracted to a single helper to avoid
        duplicating the extraction logic.
        """
        entity_value = consequences["value"].squeeze(-1)
        entity_risk = consequences["terminal_risk"].squeeze(-1)
        entity_uncert = consequences["uncertainty"].squeeze(-1)
        return entity_value - entity_risk - entity_uncert

    def _consequence_grounding_confidence(self):
        """Continuous evidence mass for learned consequence control."""
        count = float(self.consequence_events_seen.item())
        return count / (count + 10.0)

    def _future_grounding_confidence(self):
        if self.use_prospective_event_grounding:
            count = float(self.prospective_events_seen.item())
            return count / (count + 10.0)
        return self._consequence_grounding_confidence()

    def select_action(self, obs, evaluate=False):
        """Steps the OLF organism forward through the full organismic loop."""
        if self.ghost is not None and self.ghost.has_pending_release:
            raise RuntimeError(
                "external consequence must recouple the pending ghost release "
                "before another action is selected"
            )
        device = next(self.parameters()).device
        obs_t = torch.FloatTensor(obs).to(device).unsqueeze(0)

        # Parse observations
        if obs_t.shape[-1] == self.obs_dim:
            agent_pos, self_state, context, entities_pos, entities_feats = self.parse_obs(obs_t)
        else:
            # Defensive fallback: pad with zeros up to obs_dim
            pad = torch.zeros(1, self.obs_dim - obs_t.shape[-1], device=device)
            obs_t = torch.cat([obs_t, pad], dim=-1)
            agent_pos, self_state, context, entities_pos, entities_feats = self.parse_obs(obs_t)

        # 1. Retrieve SPM temporal trace
        spm_trace = self.spm.get_trace().to(device)

        # 2. Update continuous flow h on sphere S^(d-1)
        # ``learn_consequence(next_obs=...)`` may already have recoupled this
        # exact observation. Reusing it once avoids integrating the same sensory
        # consequence twice.
        prefetched = self._prefetched_obs
        if prefetched is not None and torch.equal(obs_t, prefetched.to(device)):
            self._prefetched_obs = None
        else:
            self._prefetched_obs = None
            x_flow = torch.cat([obs_t, spm_trace], dim=-1)
            self.h = self.flow(x_flow, self.h.to(device))

        # The attractor field is maintained as a
        # diagnostic mechanism (which regions of latent space are
        # preferred). Per the brief, goals are attractors, not commands.
        # a small attractor bias (0.001) applies only
        # after the organism has accumulated some experience
        # (`self._steps_seen > 50`). This lets the attractor guide
        # long-horizon behavior without destabilizing early training.
        if hasattr(self, "_steps_seen"):
            self._steps_seen += 1
        else:
            self._steps_seen = 1
        if self._steps_seen > 50:
            with torch.no_grad():
                tendency, _bias = self.attractor_field.compute_tendency(self.h, dt=0.01)
                if tendency.shape != self.h.shape:
                    tendency = tendency.reshape(self.h.shape)
                blended = 0.999 * self.h + 0.001 * tendency
                self.h = F.normalize(blended, p=2, dim=-1)

        # This is the situated latent from which the released action actually
        # departs. The old code snapshotted before perceiving obs_t.
        self._h_at_action = self.h.clone().detach()
        self._entity_feats_at_action = entities_feats.detach().clone()

        # transport persistent ghosts to the current real anchor. Runs
        # only when a ghost subsystem exists; purely detached memory update.
        if self.ghost is not None:
            with torch.no_grad():
                self.ghost.begin_step(self.h)

        # 3. Situated Semantics Binding
        sigma_t = self.semantics.bind(spm_trace, entities_pos, entities_feats, context, self_state)

        # diagnostic: σ self-state-flip test. Bind twice with
        # self_state flipped in [-1, 1] bounds (no autograd, no
        # behavior change) and measure the L2 distance between the
        # two bound sigmas. This is the constitutional measure of
        # whether FiLM is making self_state causally active.
        if self.diag_mode:
            with torch.no_grad():
                self_state_flip = 1.0 - self_state
                sigma_flip = self.semantics.bind(
                    spm_trace, entities_pos, entities_feats, context, self_state_flip
                )
                sigma_diff = (sigma_t - sigma_flip).reshape(-1).norm().item()
                # FiLM parameter norms + gamma/beta stats
                film_w_norm = float(self.semantics.film_gen.weight.norm().item())
                film_b_norm = float(self.semantics.film_gen.bias.norm().item())
                gb = self.semantics.film_gen(self_state)
                gamma_raw, beta = gb.chunk(2, dim=-1)
                gamma_norm = float(gamma_raw.norm().item())
                beta_norm = float(beta.norm().item())
                self.diag_buffer.append({
                    "sigma_flip_l2": sigma_diff,
                    "film_weight_norm": film_w_norm,
                    "film_bias_norm": film_b_norm,
                    "film_gamma_raw_norm": gamma_norm,
                    "film_beta_norm": beta_norm,
                })

        # Form the mean of the organism's stochastic abstract action policy.
        # Exploration is sampled only after the deterministic OLF corrections,
        # so the recorded score is the score of the process that generated the
        # downstream candidate action.
        # Meaning is learned from observed consequences, not from whichever
        # policy-gradient direction happened to improve an episode. Motor/FLC
        # consume the representation without owning the semantic encoder.
        policy_sigma = sigma_t.detach()
        flat_embeds = policy_sigma.reshape(1, -1)
        policy_inputs = torch.cat([self.h, flat_embeds], dim=-1)
        action_mean = self.movement_policy(policy_inputs)

        # FLC is an OLF subsystem: current latent -> future latent -> inverse
        # transfer correction -> action-policy mean. Stochastic exploration is
        # applied after this correction; boundary and motor remain separate
        # downstream stages.
        action_mean, flc_diag = self.apply_future_control(
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

        # 4. Impasse and context assessment
        physical_impasse = self.impasse_detector.detect()
        ghost_tension = None
        dialectical_pressure = 0.0
        if (
            self.ghost is not None
            and self.ghost.config.influences_action
        ):
            ghost_tension = self.ghost.tension()
            if ghost_tension["defined"]:
                dialectical_pressure = float(ghost_tension["normalized"])
        impasse_pressure = max(float(physical_impasse), dialectical_pressure)
        impasse_detected = impasse_pressure > 0.0
        grounding_confidence = self._consequence_grounding_confidence()
        predicted_uncertainty = consequences["uncertainty"].mean().item()
        uncertainty = (
            (1.0 - grounding_confidence)
            + grounding_confidence * predicted_uncertainty
        )
        risk = (
            grounding_confidence
            * consequences["terminal_risk"].max().item()
        )
        closure_pressure = self_state.squeeze(0).cpu().numpy()

        # Compute need_pressure from self_state.
        # Need is not danger — it drives urgency, not rollback.
        # need_pressure = max(hunger, fatigue): the dominant unresolved need.
        with torch.no_grad():
            need_pressure = float(self_state.squeeze(0).max().item())

        # Compute affordance_pressure from consequence predictions.
        # Per-entity: affordance_i = value_i - risk_i - uncertainty_i
        # Scalar summary: mean over entities.
        with torch.no_grad():
            entity_affordance = self._compute_entity_affordance(consequences)
            affordance_pressure = (
                grounding_confidence
                * float(entity_affordance.mean().item())
            )

        # diagnostic: record impasse state and pre-arbitration
        # uncertainty/risk for the diagnostic report. Logging only.
        if self.diag_mode:
            def _diag_scalar(name):
                value = flc_diag.get(name)
                if value is None:
                    return 0.0
                return float(value.reshape(-1)[0].item())

            self.diag_buffer.append({
                "step": int(self.diag_episode),
                "impasse": bool(impasse_detected),
                "physical_impasse": bool(physical_impasse),
                "dialectical_pressure": dialectical_pressure,
                "uncertainty": float(uncertainty),
                "pred_risk": float(risk),
                "consequence_grounding_confidence": grounding_confidence,
                "need_pressure": need_pressure,
                "closure_pressure": [float(x) for x in closure_pressure],
                "affordance_pressure": affordance_pressure,
                "future_horizon": _diag_scalar("future_horizon"),
                "future_abstraction": _diag_scalar("future_abstraction"),
                "future_alignment": _diag_scalar("future_alignment"),
                "flc_correction_norm": _diag_scalar("flc_correction_norm"),
                "flc_action_delta_norm": _diag_scalar("flc_action_delta_norm"),
                "flc_gain": _diag_scalar("flc_gain"),
            })
            self.diag_episode += 1

        # Get diagnostic decay (kept as separate signal, NOT conflated with closure)
        diag_decay = self.diagnostic_tracker.get_decay()

        # 5. Safety steering & action release constraint
        veto_verdict = self.prev_veto_verdict

        # Compute readiness (separate from action pressure)
        # readiness uses need_pressure (urgency from self_state),
        # NOT risk (which is terminal_risk from the Veto). Need is pressure,
        # not danger.
        readiness = self.readiness_gate.compute_readiness(
            veto_verdict, need_pressure, affordance_pressure=affordance_pressure
        )

        # if recoupling is required, reduce readiness to bias
        # mode selection toward Inspect/Recouple. This is a within-regime
        # modulation — it does not replace the verdict cascade.
        if self._recouple_required:
            readiness = readiness * 0.3

        # 6. Mode arbitration
        mode, mode_probs = self.arbitrator(
            self.h, uncertainty, impasse_pressure, risk, closure_pressure,
            self.recent_consequence_val,
            diagnostic_decay=diag_decay,
            readiness=readiness,
            veto_verdict=veto_verdict,
            recoupling_required=self._recouple_required
        )

        # Update diagnostic decay tracker
        self.diagnostic_tracker.update(uncertainty, mode.item())

        # 7. Invention generation under impasse.
        # when diag_mode is on, attach the organism's diag_buffer
        # to the motor_memory as diag_log_target so per-query telemetry
        # is collected during the invention call.
        if mode[0] == 2:
            if self.diag_mode:
                self.motor_memory.diag_log_target = self.diag_buffer
            # track motor memory retrieval for causal influence test
            self._mm_retrieved = (self.motor_memory is not None and self.motor_memory.size() > 0)
            a_cand = self.invention(self.h, sigma_t, self.consequence_memory, motor_memory=self.motor_memory)
            if self.diag_mode:
                self.motor_memory.diag_log_target = None
        else:
            a_cand = a_cand_init
            self._mm_retrieved = False

        # ghost set-valued candidate correction. The ghost subsystem uses
        # the organism's OWN FLC transfer, boundary, and reachable space; the
        # organism still owns the final boundary constraint and motor release
        # below, so mode/readiness/verdict/viability remain authoritative. In
        # observe mode propose returns a zero delta (no influence) but still
        # computes the baseline future used for contrastive evidence at recouple.
        ghost_diag = None
        ghost_influenced = False
        if self.ghost is not None and self.ghost.config.active:
            ghost_delta, ghost_diag, ghost_influenced, base_future, ghost_token = (
                self.ghost.propose(
                    sigma_flat=flat_embeds,
                    self_state=self_state,
                    sigma_t=sigma_t,
                    base_action=a_cand,
                )
            )
            if ghost_influenced and ghost_delta is not None:
                # Add the detached, non-parametric correction on top of the
                # policy candidate (preserves the policy-gradient graph of
                # a_cand_init, exactly like the veto boundary does downstream).
                a_cand = a_cand + ghost_delta
            # A pending release token must never be overwritten by a later
            # (non-recoupled) selection. The passive baseline is recorded only
            # when no pending release already owns one.
            if ghost_token is not None and self._ghost_token is None:
                self._ghost_token = ghost_token
                self._ghost_base_future = base_future
            elif self._ghost_token is None:
                self._ghost_base_future = base_future
            ghost_diag = dict(ghost_diag or {})
            ghost_diag["ghost_influenced"] = bool(ghost_influenced)

        # diagnostic: record invention path activation and
        # candidate action norms (logging only).
        # add motor memory retrieval telemetry.
        if self.diag_mode:
            self.diag_buffer.append({
                "invention_invoked": bool(mode[0] == 2),
                "a_cand_norm": float(a_cand.detach().norm().item()),
                "motor_memory_retrieved": self._mm_retrieved,
                "motor_memory_size": int(self.motor_memory.size()) if self.motor_memory is not None else 0,
            })

        # Re-predict consequences with actual action proposal
        consequences_final = self.semantics.predict_consequences(sigma_t, a_cand)

        # 8. Veto boundary constraint
        # Boundary Deformation Risk.
        # The Veto uses its own B_psi(h, a, dh_pred) to estimate action-attributable
        # boundary deformation risk. This is INDEPENDENT of FiLM-modulated terminal_risk.
        # Consequence model answers: "What value does this situation imply?"
        # Veto answers: "Does this movement cause irreversible boundary collapse?"
        a_steered, veto_verdict, viability, effective_threshold, danger, veto_diag = self.veto.constrain_release(
            self.h, a_cand, sigma_t, self.semantics
        )
        self.prev_veto_verdict = veto_verdict

        # diagnostic: record veto verdict and action-norms
        # before/after veto for the calibration analysis.
        # add B_psi boundary risk signals (veto_boundary_risk, risk_baseline,
        # consequence_terminal_risk, danger).
        if self.diag_mode:
            a_pre_norm = float(a_cand.detach().norm().item())
            a_post_norm = float(a_steered.detach().norm().item())
            self.diag_buffer.append({
                "veto_verdict": str(veto_verdict),
                "viability": float(viability),
                "a_pre_veto_norm": a_pre_norm,
                "a_post_veto_norm": a_post_norm,
                "action_suppression_rate": float(a_post_norm / (a_pre_norm + 1e-8)),
                "pred_risk_at_veto": float(risk),
                "need_pressure_at_veto": need_pressure,
                "danger": danger,
                "effective_threshold": effective_threshold,
                "risk_with_action": veto_diag.get("risk_with_action", 0.0),
                "veto_boundary_risk": veto_diag.get("veto_boundary_risk", 0.0),
                "risk_baseline": veto_diag.get("risk_baseline", 0.0),
                "consequence_terminal_risk": veto_diag.get("consequence_terminal_risk", 0.0),
            })

        # Readiness-modulated motor release
        # readiness uses need_pressure, not terminal_risk.
        final_readiness = self.readiness_gate.compute_readiness(
            veto_verdict, need_pressure, affordance_pressure=affordance_pressure
        )

        # readiness_factor scales action WITHIN each verdict regime.
        # Must be computed BEFORE motor release so it can be passed through.
        readiness_factor = 0.3 if self._recouple_required else 1.0

        # 9. Motor release arbitration (readiness-scaled within regime)
        act_np = a_steered.squeeze(0).cpu().detach().numpy()
        a_final = self.motor.process_release(act_np, veto_verdict, mode.item(), viability, readiness_factor)

        # diagnostic: record motor release action and suppression.
        # add readiness scaling diagnostics.
        if self.diag_mode:
            a_pre_readiness_norm = float(np.linalg.norm(a_steered.squeeze(0).cpu().detach().numpy()))
            a_post_readiness_norm = float(np.linalg.norm(a_final))
            self.diag_buffer.append({
                "a_final_norm": a_post_readiness_norm,
                "a_final_is_zero": bool(np.allclose(a_final, 0.0)),
                "readiness_factor": float(readiness_factor),
                "a_pre_readiness_norm": a_pre_readiness_norm,
                "a_post_readiness_norm": a_post_readiness_norm,
                "readiness_scaling": float(a_post_readiness_norm / (a_pre_readiness_norm + 1e-8)),
                "recouple_required": self._recouple_required,
            })

        # 10. State trace updates
        # RTCM gets the (h_t, a_t, h_{t+1}, consequence) transition so its R_Δ
        # operator can learn the cause→effect direction.
        # The current step transition is logged. The consequence is filled in
        # retroactively by learn_consequence (training loop), and the R_Δ effect
        # delta is computed from (h_next − h_t) once h_{t+1} is known next call.
        self.spm.update(self.h)
        self.rtcm.add_step(
            self.h,
            a_final,
            consequence=None,  # filled by learn_consequence via _backfill_consequence
            h_next=None,
        )
        self.impasse_detector.add_step(agent_pos.squeeze(0).cpu().numpy(), a_final)

        # Update readiness gate with flow state
        self.readiness_gate.update(self.h, received_consequence=False)

        # Track for consequence learning
        self.last_sigma = sigma_t.clone().detach()
        self.last_action = a_final.copy()
        if self.ghost is not None and self._ghost_token is not None:
            self.ghost.finalize_release(
                self._ghost_token,
                real_prev=self._h_at_action,
                released_action=a_final,
            )
        
        # set recouple obligation when verdict demands it
        if veto_verdict in ("recouple", "rollback"):
            self._recouple_required = True
        
        return a_final, {
            "mode": mode.item(),
            "mode_probs": mode_probs.squeeze(0).cpu().detach().numpy(),
            "consequences": consequences_final,
            "impasse": impasse_detected,
            "impasse_pressure": impasse_pressure,
            "ghost_tension": ghost_tension,
            "risk": risk,
            "need_pressure": need_pressure,
            "danger": danger,
            "effective_threshold": effective_threshold,
            "verdict": veto_verdict,
            "viability": float(viability),
            "readiness": final_readiness,
            "readiness_factor": float(readiness_factor),
            "affordance_pressure": affordance_pressure,
            "consequence_grounding_confidence": grounding_confidence,
            "diagnostic_decay": diag_decay,
            "recouple_required": self._recouple_required,
            "ghost": ghost_diag,
            "future_horizon": float(flc_diag["future_horizon"].reshape(-1)[0].item()),
            "future_abstraction": float(flc_diag["future_abstraction"].reshape(-1)[0].item()),
            "future_alignment": float(flc_diag["future_alignment"].reshape(-1)[0].item()),
            "flc_correction_norm": float(flc_diag["flc_correction_norm"].reshape(-1)[0].item()),
            "flc_action_delta_norm": float(flc_diag["flc_action_delta_norm"].reshape(-1)[0].item()),
            "flc_gain": float(flc_diag["flc_gain"].reshape(-1)[0].item()),
            "future_hint_confidence": float(
                flc_diag["future_hint_confidence"].reshape(-1)[0].item()
            ),
            "memory_action_confidence": float(
                flc_diag["memory_action_confidence"].reshape(-1)[0].item()
            ),
            "risk_with_action": veto_diag.get("risk_with_action", 0.0),
            "veto_boundary_risk": veto_diag.get("veto_boundary_risk", 0.0),
            "risk_baseline": veto_diag.get("risk_baseline", 0.0),
            # Internal rollout trace consumed by experiments.run_core. The
            # log-probability retains its graph; diagnostic tensors do not.
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
        """Recouple to an outcome and optionally store persistent traces.

        The trace follows (situation_before, action, situation_after, consequence).
        - situation_before: snapshot of h BEFORE the action that produced this consequence
        - situation_after: current h (the state that resulted)
        - action: the action that bridged before and after
        - consequence: [reward, was_lethal, hunger_delta, fatigue_delta]

        Also backfills the most recent RTCM step's consequence so that R_Δ
        training sees the actual per-step consequence (not just the final
        episode outcome).
        """
        # diagnostic: record consequence signal and any
        # MotorMemory write. Logging only.
        if self.diag_mode:
            self.diag_buffer.append({
                "consequence": {
                    "reward": float(reward),
                    "was_lethal": float(was_lethal),
                    "hunger_delta": float(hunger_delta),
                    "fatigue_delta": float(fatigue_delta),
                },
                "motor_memory_size": int(self.motor_memory.size()),
                "recouple_cleared": self._recouple_required,
                "persistent_store": bool(store),
            })

        if self.last_sigma is not None and self.last_action is not None:
            consequence_vec = np.array(
                [reward, float(was_lethal), hunger_delta, fatigue_delta],
                dtype=np.float32,
            )
            ghost_context = None
            if self.ghost is not None and self._ghost_token is not None:
                ghost_context = self.ghost.pending_context(self._ghost_token)
            s_t = (
                ghost_context["real_prev"]
                if ghost_context is not None
                else self._h_at_action if self._h_at_action is not None else self.h
            )

            # Recouple through the actual next observation before writing any
            # transformation memory. Legacy callers may omit next_obs; they keep
            # the old no-deformation fallback for API compatibility only.
            s_after = self._recouple_observation(
                next_obs,
                s_t=s_t,
                track_grad=bool(store and self.training),
            )
            self.last_observed_effect = log_map_sphere(
                s_t.detach(), s_after.detach()
            )

            # ghost external recoupling. The pending-release token issued
            # by an influenced select_action is consumed exactly once here; an
            # arbitrary latent cannot satisfy recoupling. In observe mode the
            # token is None and evidence is still updated (detached).
            if self.ghost is not None and self.ghost.config.active:
                with torch.no_grad():
                    ghost_report = self.ghost.recouple_token(
                        token=self._ghost_token,
                        real_prev=s_t.detach(),
                        observed_anchor=s_after.detach(),
                        base_future_anchor=(
                            self._ghost_base_future.detach()
                            if self._ghost_base_future is not None
                            else s_t.detach()
                        ),
                        released_action=self.last_action,
                    )
                if ghost_report.get("updated", False):
                    self._ghost_token = None
                    self._ghost_base_future = None

            event_mask = torch.zeros(
                self.num_entities, dtype=torch.bool, device=self.h.device
            )
            if (
                self._entity_feats_at_action is not None
                and self._prefetched_obs is not None
            ):
                _, _, _, _, next_entity_feats = self.parse_obs(
                    self._prefetched_obs.to(self.h.device)
                )
                event_mask = entity_feature_event_mask(
                    self._entity_feats_at_action.to(self.h.device),
                    next_entity_feats,
                ).squeeze(0)
            self.last_entity_event_mask = event_mask.detach()
            if store:
                self.consequence_events_seen.add_(event_mask.sum())
                if self.use_prospective_event_grounding:
                    self.prospective_events_seen.add_(event_mask.sum())

            if store:
                self.consequence_memory.add_trace(
                    self.last_sigma.mean(dim=1),
                    self.last_action,
                    s_t,
                    consequence_vec,
                    s_after=s_after,
                )
            self.rtcm.complete_last_step(consequence_vec, s_after)

            # Arbitration receives only organism-native deformation: relief is
            # positive, increasing need and lethal collapse are negative. Raw
            # benchmark reward must not become an implicit mode label.
            self.recent_consequence_val = (
                -float(hunger_delta)
                - float(fatigue_delta)
                - float(was_lethal)
            )

            # body relief can establish a preferred latent
            # region. This uses observed homeostatic deformation, never a task
            # reward or status label. Bounded attractor capacity prevents
            # unbounded writes; prospective consolidation is handled separately.
            if store and hasattr(self, "attractor_field") and not (
                hasattr(self, "ablation_type") and self.ablation_type == "no_closure_pressure"
            ):
                body_relief = max(0.0, -float(hunger_delta)) + max(
                    0.0, -float(fatigue_delta)
                )
                if body_relief > 1e-6 and not was_lethal:
                    with torch.no_grad():
                        self.attractor_field.create_at(s_after)

            # record the (before_h, action, after_h) transformation into
            # motor_memory. The success flag is 0.0 if lethal, 1.0 otherwise.
            # Reward is the raw reward signal (used for ranking similar
            # transformations in query_similar_action).
            if store and hasattr(self, "motor_memory") and not (
                hasattr(self, "ablation_type") and self.ablation_type == "no_motor_memory"
            ):
                success_flag = 0.0 if was_lethal else 1.0
                self.motor_memory.add_transformation(
                    s_t,
                    self.last_action,
                    s_after,
                    success_flag,
                    float(reward),
                )

            # Mark that consequence was received (recoupling)
            # clear recouple obligation — the next real observation
            # satisfies the world-mutating action's recoupling requirement.
            self._recouple_required = False
            self.readiness_gate.update(self.h, received_consequence=True)

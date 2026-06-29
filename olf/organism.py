import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    def __init__(self, obs_dim=18, action_dim=3, latent_dim=32, hidden_dim=64):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
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

        # v3: motor transformation memory. Stores (before_h, action, after_h,
        # success, reward) with explicit delta. InventionGenerator queries
        # this to compose candidate action sequences from previously-observed
        # successful transformations, rather than just replaying old actions.
        from olf.motor_memory import MotorMemory
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

        # Constitution §12: Readiness gate
        self.readiness_gate = ReadinessGate()

        # Constitution §13: Diagnostic decay tracker
        self.diagnostic_tracker = DiagnosticDecayTracker()

        # Constitution §6: Attractor field. Goals are attractors in
        # latent space, not symbolic commands.
        from olf.attractor import AttractorField
        self.attractor_field = AttractorField(latent_dim=latent_dim)

        # Action-Sphere RTCM memo §8: Prospective salience gate. Decides
        # what events to write into long-term memory based on estimated
        # future causal value, not just because they happened.
        from olf.salience import ProspectiveSalienceGate
        self.salience_gate = ProspectiveSalienceGate(latent_dim=latent_dim)

        # Policy model for physical moves (dx, dy, u)
        # Constitution §6: policy emerges from the latent flow, but for the
        # v0.1 prototype we use a small MLP head that maps the bound sigma
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

        # v0.3.2: stateful recoupling obligation. When a world-mutating
        # action is released, recouple_required becomes True. The next
        # real observation/consequence satisfies the obligation, and
        # readiness is restored. Do NOT fake perception inside select_action.
        self._recouple_required = False

        # Last-mode memory: carries across select_action calls until motor release
        self.prev_veto_verdict = "release"

        # Snapshot of h at the start of select_action (for honest s_{t+1} in trace)
        self._h_at_action = None

        # v0.3.1.2: diagnostic instrumentation. When diag_mode is True, the
        # organism records per-step telemetry into diag_buffer for the
        # diagnostic report. Forward pass, gradient flow, and any
        # trainable parameters are unchanged. Default is off.
        self.diag_mode: bool = False
        self.diag_buffer: list = []
        self.diag_episode: int = 0

        self.reset_state()
        
    def reset_diag(self):
        """v0.3.1.2: reset the diagnostic buffer. Called by the diagnostic
        runner between episodes. Telemetry only — no behavior change.
        """
        self.diag_buffer = []
        self.diag_episode = 0
        # Reset per-call counters on child modules
        if hasattr(self, "motor_memory"):
            self.motor_memory._query_count = 0

    def self_state_swap_probe(self, obs):
        """v0.3.2.5: measure whether self_state changes downstream behavior.

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

            # v0.3.2.6: compute affordance_pressure for both self_states
            af_orig = float((cons_orig["value"].squeeze(-1) - cons_orig["terminal_risk"].squeeze(-1) - cons_orig["uncertainty"].squeeze(-1)).mean().item())
            af_flip = float((cons_flip["value"].squeeze(-1) - cons_flip["terminal_risk"].squeeze(-1) - cons_flip["uncertainty"].squeeze(-1)).mean().item())
            affordance_pressure_diff = float(abs(af_orig - af_flip))

            # Mode difference: run arbitrator with both consequence sets
            impasse = self.impasse_detector.detect()
            uncertainty_o = cons_orig["uncertainty"].mean().item()
            risk_o = cons_orig["terminal_risk"].max().item()
            closure_p = self_state.squeeze(0).cpu().numpy()
            diag_decay = self.diagnostic_tracker.get_decay()
            # v0.3.2.8: need_pressure for readiness (not risk)
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
            # v0.3.2.8: need_pressure for readiness (not risk)
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
        # Initialize h as a proper random unit vector on S^(d-1) (Constitution §2)
        h_init = torch.randn(1, self.latent_dim)
        self.h = h_init / (torch.linalg.norm(h_init, dim=-1, keepdim=True) + 1e-8)
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
        self.last_h_at_action = None  # s_t snapshot for honest (s_t, a_t, s_{t+1}) trace
        self.recent_consequence_val = 0.0

        # The veto starts the episode with a clean slate
        self.prev_veto_verdict = "release"
        self._recouple_required = False
        
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

    def apply_future_control(self, sigma_t, self_state, base_action):
        """Apply OLF's FLC subsystem to a base action proposal."""
        sigma_flat = sigma_t.reshape(base_action.shape[0], -1)
        return self.flc(self.h, sigma_flat, self_state, base_action)

    def select_action(self, obs, evaluate=False):
        """Steps the OLF organism forward through the full organismic loop."""
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

        # Snapshot h BEFORE the flow update: this is s_t for the (s_t, a_t, s_{t+1}) trace
        self._h_at_action = self.h.clone().detach()

        # 1. Retrieve SPM temporal trace
        spm_trace = self.spm.get_trace().to(device)

        # 2. Update continuous flow h on sphere S^(d-1)
        x_flow = torch.cat([obs_t, spm_trace], dim=-1)
        self.h = self.flow(x_flow, self.h.to(device))

        # Constitution §6: The attractor field is maintained as a
        # diagnostic mechanism (which regions of latent space are
        # preferred). Per the brief, goals are attractors, not commands.
        # v3.1.1: we apply a very small attractor bias (0.001) but only
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

        # 3. Situated Semantics Binding (Constitution §3)
        sigma_t = self.semantics.bind(spm_trace, entities_pos, entities_feats, context, self_state)

        # v0.3.1.2 diagnostic: σ self-state-flip test. Bind twice with
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
                # v0.3.2.5: FiLM parameter norms + gamma/beta stats
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

        # Generate action proposal from movement policy
        flat_embeds = sigma_t.reshape(1, -1)
        policy_inputs = torch.cat([self.h, flat_embeds], dim=-1)
        a_move = self.movement_policy(policy_inputs)  # (1, 3): (dx, dy, u)

        # Add exploration noise during training (not evaluation)
        if not evaluate:
            decay = max(0.0, 1.0 - self.episode_count / (self.warmup_episodes * 5))
            noise_std = self.explore_noise_min + (self.explore_noise_init - self.explore_noise_min) * decay
            noise = torch.randn_like(a_move) * noise_std
            a_move = torch.clamp(a_move + noise, -1.0, 1.0)

        # Candidate action assembly: a_move now already contains (dx, dy, u) — no padding needed.
        a_cand_init = a_move
        consequences = self.semantics.predict_consequences(sigma_t, a_cand_init)

        # v0.3.2.6: Per-entity affordance gradient → action pressure.
        # Shift action toward the entity with highest affordance. No learned
        # parameters — uses the value predictions directly. This is the
        # "entity consequence gradient → action pressure" path.
        with torch.no_grad():
            entity_value = consequences["value"].squeeze(-1)
            entity_risk = consequences["terminal_risk"].squeeze(-1)
            entity_uncert = consequences["uncertainty"].squeeze(-1)
            entity_affordance = entity_value - entity_risk - entity_uncert
            afford_weights = torch.softmax(entity_affordance, dim=-1)
            afford_dir = (afford_weights.unsqueeze(-1) * entities_pos).sum(dim=1)
            afford_dir = F.normalize(afford_dir + 1e-8, p=2, dim=-1)
            af_mag = max(-0.3, min(0.3, entity_affordance.mean().item()))
            a_cand_init = a_cand_init.clone()
            a_cand_init[:, :2] = a_cand_init[:, :2] + 0.2 * af_mag * afford_dir

        # FLC is an OLF subsystem: current latent -> future latent -> inverse
        # transfer correction -> action proposal. Boundary and motor remain
        # separate downstream stages.
        a_cand_init, flc_diag = self.apply_future_control(
            sigma_t, self_state, a_cand_init
        )
        consequences = self.semantics.predict_consequences(sigma_t, a_cand_init)

        # 4. Impasse and context assessment
        impasse_detected = self.impasse_detector.detect()
        uncertainty = consequences["uncertainty"].mean().item()
        risk = consequences["terminal_risk"].max().item()
        closure_pressure = self_state.squeeze(0).cpu().numpy()

        # v0.3.2.9: Compute need_pressure from self_state.
        # Need is not danger — it drives urgency, not rollback.
        # need_pressure = max(hunger, fatigue): the dominant unresolved need.
        with torch.no_grad():
            need_pressure = float(self_state.squeeze(0).max().item())

        # v0.3.2.6: Compute affordance_pressure from consequence predictions.
        # Per-entity: affordance_i = value_i - risk_i - uncertainty_i
        # Scalar summary: mean over entities.
        with torch.no_grad():
            entity_value = consequences["value"].squeeze(-1)
            entity_risk = consequences["terminal_risk"].squeeze(-1)
            entity_uncert = consequences["uncertainty"].squeeze(-1)
            entity_affordance = entity_value - entity_risk - entity_uncert
            affordance_pressure = float(entity_affordance.mean().item())

        # v0.3.1.2 diagnostic: record impasse state and pre-arbitration
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
                "uncertainty": float(uncertainty),
                "pred_risk": float(risk),
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

        # Constitution §13: Get diagnostic decay (kept as separate signal, NOT conflated with closure)
        diag_decay = self.diagnostic_tracker.get_decay()

        # 5. Safety steering & action release constraint (Constitution §7)
        veto_verdict = self.prev_veto_verdict

        # Constitution §12: Compute readiness (separate from action pressure)
        # v0.3.2.8: readiness uses need_pressure (urgency from self_state),
        # NOT risk (which is terminal_risk from the Veto). Need is pressure,
        # not danger.
        readiness = self.readiness_gate.compute_readiness(
            veto_verdict, need_pressure, affordance_pressure=affordance_pressure
        )

        # v0.3.2.2: if recoupling is required, reduce readiness to bias
        # mode selection toward Inspect/Recouple. This is a within-regime
        # modulation — it does not replace the verdict cascade.
        if self._recouple_required:
            readiness = readiness * 0.3

        # 6. Mode arbitration (Constitution §11: soft biases, not hard overrides)
        mode, mode_probs = self.arbitrator(
            self.h, uncertainty, impasse_detected, risk, closure_pressure,
            self.recent_consequence_val,
            diagnostic_decay=diag_decay,
            readiness=readiness,
            veto_verdict=veto_verdict,
            recoupling_required=self._recouple_required
        )

        # Constitution §13: Update diagnostic decay tracker
        self.diagnostic_tracker.update(uncertainty, mode.item())

        # 7. Invention generation (under impasse, Constitution §10)
        # v0.3.1.2: when diag_mode is on, attach the organism's diag_buffer
        # to the motor_memory as diag_log_target so per-query telemetry
        # is collected during the invention call.
        if mode[0] == 2:
            if self.diag_mode:
                self.motor_memory.diag_log_target = self.diag_buffer
            # v0.3.2: track motor memory retrieval for causal influence test
            self._mm_retrieved = (self.motor_memory is not None and self.motor_memory.size() > 0)
            a_cand = self.invention(self.h, sigma_t, self.consequence_memory, motor_memory=self.motor_memory)
            if self.diag_mode:
                self.motor_memory.diag_log_target = None
        else:
            a_cand = a_cand_init
            self._mm_retrieved = False

        # v0.3.1.2 diagnostic: record invention path activation and
        # candidate action norms (logging only).
        # v0.3.2: add motor memory retrieval telemetry.
        if self.diag_mode:
            self.diag_buffer.append({
                "invention_invoked": bool(mode[0] == 2),
                "a_cand_norm": float(a_cand.detach().norm().item()),
                "motor_memory_retrieved": self._mm_retrieved,
                "motor_memory_size": int(self.motor_memory.size()) if self.motor_memory is not None else 0,
            })

        # Re-predict consequences with actual action proposal
        consequences_final = self.semantics.predict_consequences(sigma_t, a_cand)

        # 8. Veto boundary constraint (Constitution §7: viability pressure, not binary)
        # v0.3.2.10 — Boundary Deformation Risk.
        # The Veto uses its own B_psi(h, a, dh_pred) to estimate action-attributable
        # boundary deformation risk. This is INDEPENDENT of FiLM-modulated terminal_risk.
        # Consequence model answers: "What value does this situation imply?"
        # Veto answers: "Does this movement cause irreversible boundary collapse?"
        a_steered, veto_verdict, viability, effective_threshold, danger, veto_diag = self.veto.constrain_release(
            self.h, a_cand, sigma_t, self.semantics
        )
        self.prev_veto_verdict = veto_verdict

        # v0.3.1.2 diagnostic: record veto verdict and action-norms
        # before/after veto for the calibration analysis.
        # v0.3.2.10: add B_psi boundary risk signals (veto_boundary_risk, risk_baseline,
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

        # Constitution §12: Readiness-modulated motor release
        # v0.3.2.9: readiness uses need_pressure, not terminal_risk.
        final_readiness = self.readiness_gate.compute_readiness(
            veto_verdict, need_pressure, affordance_pressure=affordance_pressure
        )

        # v0.3.2.2: readiness_factor scales action WITHIN each verdict regime.
        # Must be computed BEFORE motor release so it can be passed through.
        readiness_factor = 0.3 if self._recouple_required else 1.0

        # 9. Motor release arbitration (v0.3.2.2: readiness-scaled within regime)
        act_np = a_steered.squeeze(0).cpu().detach().numpy()
        a_final = self.motor.process_release(act_np, veto_verdict, mode.item(), viability, readiness_factor)

        # v0.3.1.2 diagnostic: record motor release action and suppression.
        # v0.3.2.2: add readiness scaling diagnostics.
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
        # We log the current step's transition. The consequence is filled in
        # retroactively by learn_consequence (training loop), and the R_Δ effect
        # delta is computed from (h_next − h_t) once h_{t+1} is known next call.
        self.spm.update(self.h)
        self.rtcm.add_step(
            self._h_at_action if self._h_at_action is not None else self.h,
            a_final,
            consequence=None,  # filled by learn_consequence via _backfill_consequence
            h_next=self.h,
        )
        self.impasse_detector.add_step(agent_pos.squeeze(0).cpu().numpy(), a_final)

        # Constitution §12: Update readiness gate with flow state
        self.readiness_gate.update(self.h, received_consequence=False)

        # Track for consequence learning
        self.last_sigma = sigma_t.clone().detach()
        self.last_action = a_final.copy()
        
        # v0.3.2: set recouple obligation when verdict demands it
        if veto_verdict in ("recouple", "rollback"):
            self._recouple_required = True
        
        return a_final, {
            "mode": mode.item(),
            "mode_probs": mode_probs.squeeze(0).cpu().detach().numpy(),
            "consequences": consequences_final,
            "impasse": impasse_detected,
            "risk": risk,
            "need_pressure": need_pressure,
            "danger": danger,
            "effective_threshold": effective_threshold,
            "verdict": veto_verdict,
            "viability": float(viability),
            "readiness": final_readiness,
            "readiness_factor": float(readiness_factor),
            "affordance_pressure": affordance_pressure,
            "diagnostic_decay": diag_decay,
            "recouple_required": self._recouple_required,
            "future_horizon": float(flc_diag["future_horizon"].reshape(-1)[0].item()),
            "future_abstraction": float(flc_diag["future_abstraction"].reshape(-1)[0].item()),
            "future_alignment": float(flc_diag["future_alignment"].reshape(-1)[0].item()),
            "flc_correction_norm": float(flc_diag["flc_correction_norm"].reshape(-1)[0].item()),
            "flc_action_delta_norm": float(flc_diag["flc_action_delta_norm"].reshape(-1)[0].item()),
            "risk_with_action": veto_diag.get("risk_with_action", 0.0),
            "veto_boundary_risk": veto_diag.get("veto_boundary_risk", 0.0),
            "risk_baseline": veto_diag.get("risk_baseline", 0.0),
        }
        
    def learn_consequence(self, reward, was_lethal, hunger_delta, fatigue_delta):
        """Stores step outcome in consequence trace memory buffer.

        The trace follows Constitution §4: (situation_before, action, situation_after, consequence).
        - situation_before: snapshot of h BEFORE the action that produced this consequence
        - situation_after: current h (the state that resulted)
        - action: the action that bridged before and after
        - consequence: [reward, was_lethal, hunger_delta, fatigue_delta]

        Also backfills the most recent RTCM step's consequence so that R_Δ
        training sees the actual per-step consequence (not just the final
        episode outcome).
        """
        # v0.3.1.2 diagnostic: record consequence signal and any
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
            })

        if self.last_sigma is not None and self.last_action is not None:
            consequence_vec = np.array(
                [reward, float(was_lethal), hunger_delta, fatigue_delta],
                dtype=np.float32,
            )
            s_t = self._h_at_action if self._h_at_action is not None else self.h
            # Pass current h (s_{t+1}) as s_after for slow consolidation
            self.consequence_memory.add_trace(
                self.last_sigma.mean(dim=1),
                self.last_action,
                s_t,
                consequence_vec,
                s_after=self.h,
            )
            # Backfill RTCM with the per-step consequence so R_Δ learns the
            # effect actually observed at this step, not just the final outcome.
            if self.rtcm.history:
                self.rtcm.history[-1]["consequence"] = torch.FloatTensor(
                    consequence_vec
                ).unsqueeze(0)

            self.recent_consequence_val = reward - 1.0 * was_lethal

            # Constitution §6: Update the attractor field on strong
            # positive or lethal consequences. v3.1.1: throttled to once
            # per 10 episodes to avoid attractor explosion.
            if hasattr(self, "attractor_field") and not (
                hasattr(self, "ablation_type") and self.ablation_type == "no_closure_pressure"
            ):
                if not hasattr(self, "_episodes_since_attractor"):
                    self._episodes_since_attractor = 0
                self._episodes_since_attractor += 1
                if self._episodes_since_attractor >= 10:
                    self._episodes_since_attractor = 0
                    if reward > 0.5 and not was_lethal:
                        with torch.no_grad():
                            self.attractor_field.create_at(self.h)

            # v3: record the (before_h, action, after_h) transformation into
            # motor_memory. The success flag is 0.0 if lethal, 1.0 otherwise.
            # Reward is the raw reward signal (used for ranking similar
            # transformations in query_similar_action).
            if hasattr(self, "motor_memory") and not (
                hasattr(self, "ablation_type") and self.ablation_type == "no_motor_memory"
            ):
                success_flag = 0.0 if was_lethal else 1.0
                self.motor_memory.add_transformation(
                    s_t,
                    self.last_action,
                    self.h,
                    success_flag,
                    float(reward),
                )

            # Constitution §12: Mark that consequence was received (recoupling)
            # v0.3.2: clear recouple obligation — the next real observation
            # satisfies the world-mutating action's recoupling requirement.
            self._recouple_required = False
            self.readiness_gate.update(self.h, received_consequence=True)

import torch
import torch.nn as nn

class TriadicSemantics(nn.Module):
    """
    TriadicSemantics performs situated binding of temporal trace (SPM), Object, Context,
    and Self-state, then predicts the consequence profile of a candidate action.
    
    Formula:
      sigma_t = Bind(SPM_t, o_t, c_t, s_t)
      C_phi(sigma_t, a_t) -> { dh_pred, value, terminal_risk, reversibility, uncertainty }
    """
    def __init__(self, spm_dim=32, entity_dim=6, context_dim=2, self_state_dim=2,
                 action_dim=3, hidden_dim=64, latent_dim=32):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.self_state_dim = self_state_dim

        # v3: split binding into pre-binder (no self_state) + FiLM modulation.
        # Pre-binder sees [SPM, o, context] only. v3.1 fix: pre-binder is
        # 2-layer (matching the original binder's depth) so that the
        # binding is not artificially weakened.
        pre_binder_input_dim = spm_dim + entity_dim + context_dim
        self.pre_binder = nn.Sequential(
            nn.Linear(pre_binder_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # FiLM generator: self_state → (gamma_raw, beta), each of size hidden_dim.
        # Final modulation: sigma_t = (1 + gamma_raw) ⊙ pre + beta
        # v3 amendment: true identity initialization (gamma_raw=0, beta=0) so
        # at init the binder behaves exactly like a no-FiLM linear.
        self.film_gen = nn.Linear(self_state_dim, 2 * hidden_dim)
        with torch.no_grad():
            self.film_gen.weight.zero_()
            self.film_gen.bias.zero_()

        # v0.3.1.2 diagnostic: FiLM parameter norm snapshot for the
        # "is FiLM dead?" check. Logging only.
        self._initial_film_norm = float(self.film_gen.weight.norm().item())

        # Kept for backward compatibility (some legacy code may reference
        # self.binder). It is no longer used by bind() but kept as an
        # attribute on the module so checkpoints remain loadable.
        binding_input_dim = spm_dim + entity_dim + context_dim + self_state_dim
        self.binder = nn.Sequential(
            nn.Linear(binding_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Consequence prediction networks: maps (sigma_t, a_t) to outcomes
        pred_input_dim = hidden_dim + action_dim
        
        # We output:
        # - dh_pred: change in latent flow state (latent_dim)
        # - value: scalar consequence value (1)
        # - terminal_risk: scalar danger/fatality risk (1) — used by Veto (when need is low)
        # - reversibility: scalar estimation of state recovery (1)
        # - uncertainty: scalar prediction variance (1)
        output_dim = latent_dim + 1 + 1 + 1 + 1 # latent_dim + 4
        
        self.pred_l1 = nn.Linear(pred_input_dim, hidden_dim)
        self.pred_l2 = nn.Linear(hidden_dim, hidden_dim)
        self.pred_l3 = nn.Linear(hidden_dim, output_dim)
        
        # Optimistic initialization: terminal_risk starts at sigmoid(-2) ≈ 0.12
        # so the veto boundary does not block actions at initialization.
        # Output layout: [dh_pred(latent_dim), value(1), terminal_risk(1), reversibility(1), uncertainty(1)]
        with torch.no_grad():
            self.pred_l3.bias[latent_dim] = 0.5         # value → mildly positive prior
            self.pred_l3.bias[latent_dim + 1] = -2.0    # terminal_risk → sigmoid(-2) ≈ 0.12
        
    def bind(self, spm_trace, entity_rel_pos, entity_feats, context, self_state):
        """Binds SPM, Object features, Context, and Self-state into a situated
        embedding sigma. v3: self_state modulates the binding through FiLM.

        Pre-binder sees only [SPM, o, context]. FiLM from self_state produces
        (gamma_raw, beta), and the final sigma is (1 + gamma_raw) ⊙ pre + beta.
        At init, gamma_raw = 0, beta = 0 → identity modulation (sigma = pre).
        """
        num_entities = entity_rel_pos.size(1)

        # Concatenate entity rel pos and features: o_t of shape (batch, num_entities, 6)
        o_t = torch.cat([entity_rel_pos, entity_feats], dim=-1)

        # Expand spm_trace and context for each entity (self_state NOT yet).
        spm_exp = spm_trace.unsqueeze(1).expand(-1, num_entities, -1)
        ctx_exp = context.unsqueeze(1).expand(-1, num_entities, -1)

        pre_inputs = torch.cat([spm_exp, o_t, ctx_exp], dim=-1)
        pre = self.pre_binder(pre_inputs)  # (batch, num_entities, hidden_dim)

        # FiLM: self_state → (gamma_raw, beta).
        gb = self.film_gen(self_state)  # (batch, 2*hidden_dim)
        gamma_raw, beta = gb.chunk(2, dim=-1)
        gamma = 1.0 + gamma_raw  # identity at init
        sigma_t = gamma.unsqueeze(1) * pre + beta.unsqueeze(1)
        return sigma_t
        
    def predict_consequences(self, sigma_t, candidate_action):
        """
        Predicts consequence properties for a candidate action given situated embedding sigma_t.
        sigma_t: (batch, num_entities, hidden_dim)
        candidate_action: (batch, action_dim)
        """
        num_entities = sigma_t.size(1)
        
        # Expand candidate action for each entity
        action_exp = candidate_action.unsqueeze(1).expand(-1, num_entities, -1)
        
        # Concatenate: [sigma_t; a_t]
        inputs = torch.cat([sigma_t, action_exp], dim=-1)
        
        # Predict consequence profile through sequential layers
        x = torch.relu(self.pred_l1(inputs))
        x = torch.relu(self.pred_l2(x))
        raw_outputs = self.pred_l3(x) # (batch, num_entities, latent_dim + 4)
        
        # Split outputs
        dh_pred = raw_outputs[:, :, :self.latent_dim]
        consequence_value = raw_outputs[:, :, self.latent_dim:self.latent_dim+1]
        terminal_risk = torch.sigmoid(raw_outputs[:, :, self.latent_dim+1:self.latent_dim+2]) # bounded [0, 1]
        reversibility = torch.sigmoid(raw_outputs[:, :, self.latent_dim+2:self.latent_dim+3])
        uncertainty = torch.sigmoid(raw_outputs[:, :, self.latent_dim+3:self.latent_dim+4])
        
        return {
            "dh_pred": dh_pred,
            "value": consequence_value,
            "terminal_risk": terminal_risk,
            "reversibility": reversibility,
            "uncertainty": uncertainty,
        }
        
    def forward(self, spm_trace, entity_rel_pos, entity_feats, context, self_state, candidate_action):
        sigma_t = self.bind(spm_trace, entity_rel_pos, entity_feats, context, self_state)
        return self.predict_consequences(sigma_t, candidate_action), sigma_t

    def counterfactual_loss(
        self,
        spm_a, entity_a, entity_feats_a, context_a, self_state_a, action_a,
        spm_b, entity_b, entity_feats_b, context_b, self_state_b, action_b,
        target_diff,
        margin=0.5,
    ):
        """Empirical hinge-based counterfactual loss.

        Forces the consequence prediction to differ between two scenarios that
        share (object, context) but differ in self_state, and whose observed
        outcomes actually differed (target_diff is a positive scalar).

        v3 amendment: hinge loss (ReLU(margin − diff)), not a direct
        "require MSE > margin" loss. The pair is only used if target_diff > 0
        (empirical contrast), so we never invent semantic labels.

        v0.3.2: fixed entity_feats=None bug. Now accepts entity_feats_a/b.

        Returns a scalar tensor (0 if conditions not met).
        """
        if target_diff <= 0:
            return torch.tensor(0.0, device=spm_a.device)

        sigma_a = self.bind(spm_a, entity_a, entity_feats_a, context_a, self_state_a)
        sigma_b = self.bind(spm_b, entity_b, entity_feats_b, context_b, self_state_b)

        cons_a = self.predict_consequences(sigma_a, action_a)
        cons_b = self.predict_consequences(sigma_b, action_b)

        # Use the consequence "value" head as the discriminator.
        # (Other heads could be added; value is the cleanest scalar signal.)
        v_a = cons_a["value"].mean()
        v_b = cons_b["value"].mean()
        diff = torch.abs(v_a - v_b)

        # Hinge: we want diff >= target_diff + margin; loss is max(0, target_diff + margin - diff)
        return torch.clamp(target_diff + margin - diff, min=0.0)

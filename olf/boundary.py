import torch
import torch.nn as nn

class VetoBoundary(nn.Module):
    """
    VetoBoundary acts as a viability filter over actions.

    v0.3.2.10 — Boundary Deformation Risk.
    The Veto estimates action-attributable boundary deformation risk
    independently from FiLM-modulated consequence value.

    Consequence model answers: "What future state/value does this situation imply?"
    Veto answers: "Does this candidate movement deform the organism toward
    irreversible boundary collapse?"

    B_psi(h, a, dh_pred) predicts boundary risk (death, irreversible damage,
    crash, terminal collapse). This is NOT trained on reward — only on
    boundary outcomes.

    danger = max(0, B_psi(h, a, dh_pred) - B_psi(h, zero_action, dh_pred))
    Verdict cascade uses this action-attributable boundary risk.
    """
    def __init__(
        self,
        latent_dim=32,
        action_dim=3,
        threshold=0.05,
        recouple_threshold=0.10,
        rollback_threshold=0.20,
    ):
        super().__init__()
        self.threshold = threshold
        self.recouple_threshold = recouple_threshold
        self.rollback_threshold = rollback_threshold
        self.warmup = True  # Bypass FLC steering during early training

        # B_psi: predicts boundary deformation risk from (h, a, dh_pred).
        # Input: h (latent_dim) + a (action_dim) + dh_pred (latent_dim)
        # Output: boundary_risk [0, 1] — probability of irreversible collapse.
        # Trained ONLY on boundary outcomes (death, irreversible damage), NOT reward.
        self.risk_l1 = nn.Linear(latent_dim + action_dim + latent_dim, 64)
        self.risk_l2 = nn.Linear(64, 32)
        self.risk_l3 = nn.Linear(32, 1)

        # Initialize risk output bias to -2.0 so sigmoid(-2) ≈ 0.12 at init
        # (organism starts with low boundary risk, not zero — FiLM-independent)
        with torch.no_grad():
            self.risk_l3.bias.fill_(-2.0)

    def predict_risk(self, h, a, dh_pred):
        """B_psi(h, a, dh_pred) -> boundary_risk [0, 1].

        Predicts whether this (state, action, predicted_delta_h) triple
        leads to irreversible boundary collapse. Independent of FiLM.
        """
        inputs = torch.cat([h, a, dh_pred], dim=-1)
        x = torch.relu(self.risk_l1(inputs))
        x = torch.relu(self.risk_l2(x))
        return torch.sigmoid(self.risk_l3(x))

    def predict_risk_logits(self, h, a, dh_pred):
        """B_psi(h, a, dh_pred) -> raw logits (before sigmoid).

        Used for training with BCEWithLogitsLoss (more numerically stable).
        """
        inputs = torch.cat([h, a, dh_pred], dim=-1)
        x = torch.relu(self.risk_l1(inputs))
        x = torch.relu(self.risk_l2(x))
        return self.risk_l3(x)

    def constrain_release(self, h, a, sigma_t, semantics, lr=0.15, steps=5):
        """
        Boundary-aware action refinement using B_psi risk.

        The boundary refinement loop optimizes action to:
        - Maximize predicted value (from consequence model — tells "what's good")
        - Minimize action-attributable boundary risk from B_psi

        The consequence model provides value (for steering toward beneficial outcomes).
        B_psi provides boundary risk. Steering uses only the excess over
        the zero-action baseline so high need/baseline pressure is not
        mistaken for danger caused by the candidate action.
        These are separate signals answering different questions.

        Verdict cascade uses action-attributable boundary risk:
        danger = max(0, B_psi(h, a, dh_pred) - B_psi(h, zero_action, dh_pred))

        During warmup, bypasses steering entirely to let the organism explore freely.
        """
        # During warmup, pass actions through without steering
        if self.warmup:
            a_steered = torch.clamp(a.detach(), -1.0, 1.0)
            return a_steered, "release", 1.0, self.threshold, 0.0, {}

        # Set up a differentiable action reference
        a_ref = a.clone().detach().requires_grad_(True)

        # Boundary refinement: optimize action to maximize value and minimize
        # action-attributable boundary risk.
        # Value comes from consequence model (FiLM-modulated — "what's good").
        # Risk comes from B_psi (FiLM-independent — "what's irreversible").
        for _ in range(steps):
            consequences = semantics.predict_consequences(sigma_t, a_ref)
            val = consequences["value"].mean()

            # B_psi boundary risk for this candidate action
            dh_pred = consequences["dh_pred"].mean(dim=1)
            boundary_risk = self.predict_risk(h, a_ref, dh_pred)

            # Baseline pressure is not candidate-action danger. Remove it
            # inside steering, matching the verdict cascade's danger signal.
            with torch.no_grad():
                a_zero = torch.zeros_like(a_ref)
                consequences_zero = semantics.predict_consequences(sigma_t, a_zero)
                dh_pred_zero = consequences_zero["dh_pred"].mean(dim=1)
                baseline_risk = self.predict_risk(h, a_zero, dh_pred_zero)

            attributable_risk = torch.relu(boundary_risk - baseline_risk)

            # Loss: minimize -value + action-attributable boundary deformation
            loss = -val + 2.0 * attributable_risk

            # Compute gradient of loss w.r.t action
            grads = torch.autograd.grad(loss, a_ref, retain_graph=True, allow_unused=True)[0]
            if grads is None:
                break

            # Perform gradient descent
            with torch.no_grad():
                a_ref = a_ref - lr * grads
            a_ref = a_ref.clone().detach().requires_grad_(True)

        with torch.no_grad():
            a_steered = torch.clamp(a_ref, -1.0, 1.0)

            # Compute action-attributable boundary risk using B_psi
            consequences_final = semantics.predict_consequences(sigma_t, a_steered)
            dh_pred_final = consequences_final["dh_pred"].mean(dim=1)
            risk_with_action = float(self.predict_risk(h, a_steered, dh_pred_final).item())

            # Baseline: boundary risk with zero action (do nothing)
            a_zero = torch.zeros_like(a_steered)
            consequences_zero = semantics.predict_consequences(sigma_t, a_zero)
            dh_pred_zero = consequences_zero["dh_pred"].mean(dim=1)
            risk_baseline = float(self.predict_risk(h, a_zero, dh_pred_zero).item())

            # Action-attributable danger: how much boundary risk THIS action adds
            danger = max(0.0, risk_with_action - risk_baseline)

            # Verdict cascade using action-attributable boundary risk.
            # B_psi predicts absolute boundary proximity; danger is an excess
            # over zero-action baseline, so its dynamic range is much smaller
            # than raw terminal risk.
            if danger <= self.threshold:
                verdict = "release"
                viability = 1.0
            elif danger >= self.rollback_threshold:
                verdict = "rollback"
                viability = 0.0
            elif danger >= self.recouple_threshold:
                verdict = "recouple"
                viability = 0.0
            else:
                verdict = "hold"
                viability = 0.0

            # Diagnostic info for logging
            diagnostic_info = {
                "risk_with_action": risk_with_action,
                "veto_boundary_risk": risk_with_action,
                "risk_baseline": risk_baseline,
                "consequence_terminal_risk": float(consequences_final["terminal_risk"].max().item()),
            }

        return a_steered, verdict, viability, self.threshold, danger, diagnostic_info

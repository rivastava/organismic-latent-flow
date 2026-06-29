import torch
import torch.nn as nn
from olf.geometry import project_to_sphere, project_to_tangent, exponential_map

class LTCFlow(nn.Module):
    """
    LTCFlow implements the continuous-time Liquid Time-Constant (LTC) proposal,
    projected onto the tangent space of the unit sphere S^(d-1) to ensure the
    latent state h(t) evolves strictly on the compact manifold.
    """
    def __init__(self, input_dim, hidden_dim, tau_base=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_base = tau_base
        
        # Liquid time-constant parameter mapping
        self.w_tau = nn.Linear(hidden_dim, hidden_dim)
        self.u_tau = nn.Linear(input_dim, hidden_dim)
        
        # State target mapping
        self.w_a = nn.Linear(hidden_dim, hidden_dim)
        self.u_a = nn.Linear(input_dim, hidden_dim)
        
        # State update activation
        self.w_c = nn.Linear(hidden_dim, hidden_dim)
        self.u_c = nn.Linear(input_dim, hidden_dim)
        
    def flow_proposal(self, x, h):
        """
        Computes the raw continuous flow proposal u(t).
        """
        # liquid inverse time-constant: 1/tau
        inv_tau = (1.0 / self.tau_base) + torch.sigmoid(self.w_tau(h) + self.u_tau(x))
        
        # State targets and gating
        target = torch.tanh(self.w_c(h) + self.u_c(x))
        gate = torch.sigmoid(self.w_a(h) + self.u_a(x))
        
        # u(t) = - (1/tau) * h + gate * target
        u = - inv_tau * h + gate * target
        return u
        
    def step_ode(self, x, h):
        """
        Computes dh/dt = Pi_h u(t)
        """
        u = self.flow_proposal(x, h)
        return project_to_tangent(h, u)
        
    def forward(self, x, h_prev, dt=0.1, sub_steps=5):
        """
        Integrates flow on the sphere S^(d-1) over interval dt using the exponential map.
        """
        h = project_to_sphere(h_prev)
        dt_sub = dt / sub_steps
        
        for _ in range(sub_steps):
            dh_dt = self.step_ode(x, h)
            # Use geometric exponential map to move along tangent direction exactly on sphere
            h = exponential_map(h, dh_dt * dt_sub)
            
        return h

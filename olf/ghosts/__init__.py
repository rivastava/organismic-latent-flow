"""Role-free ghost population integration (experimental).

Temporary alternative spherical trajectories inside the OLF organismic loop.
See ``config``, ``integration``, ``population``, ``trajectory``, ``evidence``,
``lifecycle``, ``recoupling``, and ``diagnostics``.
"""

from .config import GhostConfig
from .trajectory import GhostTrajectory, make_ghost, transport_ghost
from .population import GhostPopulation
from .recoupling import ReachabilityBuffer, recouple, measure_reachability
from .integration import GhostIntegration

__all__ = [
    "GhostConfig",
    "GhostTrajectory",
    "make_ghost",
    "transport_ghost",
    "GhostPopulation",
    "ReachabilityBuffer",
    "recouple",
    "measure_reachability",
    "GhostIntegration",
]

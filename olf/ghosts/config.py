"""Experimental configuration for the role-free ghost population integration.

This module is intentionally free of any benchmark-specific logic, hidden
labels, reward shaping, or privileged coordinates. It only exposes the
experimental modes and ablations required by the integration contract.
"""

from dataclasses import dataclass

# Substrings that must never appear in a ghost API, attribute, or signal name.
# Ghosts are temporary alternative trajectories; they may not carry semantic,
# task, reward, relation, or world-label information of any kind.
PROHIBITED_LABEL_SUBSTRINGS = (
    "relation",
    "reward",
    "success",
    "task",
    "label",
    "env_matrix",
    "environment_matrix",
    "scripted",
    "privileged",
    "priv_coord",
    "goal",
    "world",
    "role",
    "identity",
    "meaning",
    "target_world",
    "benchmark",
)


GHOST_MODES = ("off", "observe", "influence")

ABLATIONS = (
    "no_ghosts",
    "single_ghost",
    "no_persistence",
    "centroid_before_inverse",
    "no_recoupling",
    "no_reachability",
    "random_routing",
    "no_schema_composition",
)


@dataclass
class GhostConfig:
    """Bounded experimental configuration for the ghost population.

    The production default is ``ghost_mode="off"`` which must leave all public
    OLF behavior unchanged.

    Tensor ownership (all tensors detached; ghosts are non-parametric memory):
      - latent_dim (D): sphere dimensionality
      - action_dim (A): motor dimensionality
      - capacity: hard upper bound on the number of simultaneous trajectories
      - reachability_threshold: residual scale used by reachability diagnostics
    """

    ghost_mode: str = "off"
    ablation: str | None = None

    latent_dim: int = 32
    action_dim: int = 3
    # Ghosts start empty and are born only from externally observed latent
    # deformation. There is deliberately no configured initial population.
    capacity: int = 4

    # Learned reachable tangent-deformation prototypes retained for the
    # reachability test. Reachability is action-conditioned.
    reachability_capacity: int = 16
    reachability_threshold: float = 0.5

    # Minimum evidence for refining a signature continuation with the learned
    # action-conditioned transfer map. External grounding controls participation.
    min_action_evidence: int = 2

    # Recoupling geometry step used to lift a tangent deformation to a point.
    transport_step: float = 1.0

    seed: int | None = None

    def __post_init__(self):
        if self.ghost_mode not in GHOST_MODES:
            raise ValueError(f"ghost_mode must be one of {GHOST_MODES}")
        if self.ablation is not None and self.ablation not in ABLATIONS:
            raise ValueError(f"ablation must be one of {ABLATIONS}")
        if self.capacity < 0:
            raise ValueError("capacity must be >= 0")

    @property
    def active(self) -> bool:
        """True when ghosts should run at all (off disables everything)."""
        return self.ghost_mode != "off"

    @property
    def influences_action(self) -> bool:
        """True only when ghosts may change the released action."""
        return self.ghost_mode == "influence"

    @property
    def effective_capacity(self) -> int:
        if self.ablation == "single_ghost":
            return 1
        if self.ablation == "no_ghosts":
            return 0
        return self.capacity

    @property
    def persistence_enabled(self) -> bool:
        return self.ablation != "no_persistence"

    @property
    def recoupling_enabled(self) -> bool:
        return self.ablation != "no_recoupling"

    @property
    def reachability_enabled(self) -> bool:
        return self.ablation != "no_reachability"

    @property
    def centroid_before_inverse(self) -> bool:
        return self.ablation == "centroid_before_inverse"

    @property
    def random_routing(self) -> bool:
        return self.ablation == "random_routing"

    @property
    def schema_composition_enabled(self) -> bool:
        """False only when the ``no_schema_composition`` ablation is active.

        Depth-one schema reuse (single-step ``schemas(...)``) remains active in
        every condition; only recursive composition across schema adjacencies
        (``composed_schemas(...)``) is disabled.
        """
        return self.ablation != "no_schema_composition"

"""External recoupling and the learned reachable-deformation space.

Recoupling is mandatory before the ghost population may influence another
action. It updates the real OLF flow first (handled by the caller), then
compares observed deformation with ghost predictions and only then updates
credibility/grounding and runs the lifecycle.

Reachability ownership rule: every prototype carries its SOURCE anchor,
the RELEASED ACTION that produced it, and the observed tangent deformation. A
candidate ghost is judged reachable only against prototypes whose action is
*compatible* with the candidate action (after transporting all tangents to the
current anchor). A span of unrelated past actions cannot make an incompatible
action claim the same reachable deformation. An empty buffer is reported as
unknown. Reachability is diagnostic and does not determine participation.
"""

from dataclasses import dataclass, field

import torch

from olf.geometry import (
    angular_distance,
    antipodal,
    exponential_map,
    log_map_sphere,
    parallel_transport_sphere,
    project_to_tangent,
)

from .config import GhostConfig
from .evidence import (
    baseline_error,
    observed_deformation,
    predictive_error,
    update_after_recoupling,
)
from .lifecycle import evict, maybe_birth, merge_similar, LifecycleReport
from .population import GhostPopulation


class ReachabilityBuffer:
    """Low-rank, action-conditioned union of *observed* reachable deformations.

    Each prototype is a (source_anchor, released_action, tangent) triple. No
    arbitrary attractor is injected here; prototypes come solely from
    external recoupling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.anchors: list[torch.Tensor] = []
        self.actions: list[torch.Tensor] = []
        self.prototypes: list[torch.Tensor] = []
        self.schema_ids: list[int] = []
        self.schema_support: list[float] = []
        self.schema_matches: list[int] = []
        self.segments: list[int] = []
        self.composition_positive: dict[tuple[int, ...], float] = {}
        self.composition_negative: dict[tuple[int, ...], float] = {}
        self.composition_observations: dict[tuple[int, ...], int] = {}
        self._next_schema_id = 0
        self._segment_id = 0

    def start_segment(self) -> None:
        """Prevent composition across an unobserved episode boundary."""
        if self.prototypes:
            self._segment_id += 1

    def add(self, anchor: torch.Tensor, action: torch.Tensor,
            tangent: torch.Tensor) -> None:
        if self.capacity <= 0:
            return
        self.anchors.append(anchor.detach().reshape(-1).clone())
        self.actions.append(
            action.detach().reshape(-1).to(
                device=anchor.device, dtype=anchor.dtype
            ).clone()
        )
        self.prototypes.append(
            tangent.detach().reshape(-1).to(
                device=anchor.device, dtype=anchor.dtype
            ).clone()
        )
        self.schema_ids.append(self._next_schema_id)
        self.schema_support.append(0.0)
        self.schema_matches.append(0)
        self.segments.append(self._segment_id)
        self._next_schema_id += 1
        if len(self.prototypes) > self.capacity:
            self.anchors.pop(0)
            self.actions.pop(0)
            self.prototypes.pop(0)
            self.schema_ids.pop(0)
            self.schema_support.pop(0)
            self.schema_matches.pop(0)
            self.segments.pop(0)
            self._prune_composition_evidence()

    def _prune_composition_evidence(self) -> None:
        active = set(self.schema_ids)
        stale = [
            path
            for path in self.composition_observations
            if any(schema_id not in active for schema_id in path)
        ]
        for path in stale:
            self.composition_positive.pop(path, None)
            self.composition_negative.pop(path, None)
            self.composition_observations.pop(path, None)

    def _predict_schema_path(
        self,
        path: tuple[int, ...],
        source_anchor: torch.Tensor,
        before_index: int,
    ) -> torch.Tensor | None:
        """Predict a path using only schema instances preceding its assay."""
        point = project_to_sphere_input(source_anchor)
        for schema_id in path:
            candidates = [
                index
                for index in range(before_index)
                if self.schema_ids[index] == schema_id
            ]
            if not candidates:
                return None
            representative = max(
                candidates,
                key=lambda index: (
                    self.schema_support[index],
                    self.schema_matches[index],
                    index,
                ),
            )
            if bool(antipodal(self.anchors[representative], point).any()):
                return None
            tangent = parallel_transport_sphere(
                self.anchors[representative],
                point,
                self.prototypes[representative],
            )
            point = exponential_map(point, tangent)
        return point

    def _update_composition_evidence(self, latest: int) -> None:
        """Score observed path extensions against their shorter prefixes.

        The assay is prequential: every prediction is built only from schema
        instances that occurred before the assayed path. The target is the
        externally observed endpoint of the latest recoupled transition.
        """
        segment = self.segments[latest]
        segment_start = latest
        while segment_start > 0 and self.segments[segment_start - 1] == segment:
            segment_start -= 1
        earliest = max(segment_start, latest - self.capacity + 1)
        observed_endpoint = exponential_map(
            self.anchors[latest], self.prototypes[latest]
        )
        for start in range(latest - 1, earliest - 1, -1):
            path = tuple(self.schema_ids[start : latest + 1])
            prefix = self._predict_schema_path(
                path[:-1], self.anchors[start], start
            )
            composed = self._predict_schema_path(
                path, self.anchors[start], start
            )
            if prefix is None or composed is None:
                continue
            advantage = float(
                angular_distance(prefix, observed_endpoint)
                - angular_distance(composed, observed_endpoint)
            )
            if not torch.isfinite(torch.tensor(advantage)):
                raise FloatingPointError("non-finite schema composition evidence")
            self.composition_positive[path] = (
                self.composition_positive.get(path, 0.0) + max(advantage, 0.0)
            )
            self.composition_negative[path] = (
                self.composition_negative.get(path, 0.0) + max(-advantage, 0.0)
            )
            self.composition_observations[path] = (
                self.composition_observations.get(path, 0) + 1
            )

    def consolidate_latest(self, predictive_advantage: float) -> dict:
        """Bind repeated, externally observed transformations into a schema.

        The newest recoupling may join an earlier schema only when released
        actions and observed tangent deformations have positive geometric
        coupling. Predictive advantage supplies external evidence that a ghost
        captured the deformation better than the ordinary-path baseline.
        """
        if predictive_advantage <= 0.0 or len(self.prototypes) < 2:
            return {"consolidated": False, "schema_id": None, "coupling": 0.0}

        latest = len(self.prototypes) - 1
        latest_anchor = self.anchors[latest]
        latest_action = self.actions[latest]
        latest_tangent = self.prototypes[latest]
        best_index = None
        best_coupling = 0.0
        for index in range(latest):
            action_coupling = float(torch.nn.functional.cosine_similarity(
                self.actions[index].reshape(1, -1),
                latest_action.reshape(1, -1),
                dim=-1,
            )[0])
            previous_tangent = parallel_transport_sphere(
                self.anchors[index], latest_anchor, self.prototypes[index]
            )
            deformation_coupling = float(torch.nn.functional.cosine_similarity(
                previous_tangent.reshape(1, -1),
                latest_tangent.reshape(1, -1),
                dim=-1,
            )[0])
            coupling = max(action_coupling, 0.0) * max(
                deformation_coupling, 0.0
            )
            if coupling > best_coupling:
                best_coupling = coupling
                best_index = index

        if best_index is None or best_coupling <= 0.0:
            return {"consolidated": False, "schema_id": None, "coupling": 0.0}

        schema_id = self.schema_ids[best_index]
        self.schema_ids[latest] = schema_id
        prior_indices = [
            index
            for index, candidate_id in enumerate(self.schema_ids[:-1])
            if candidate_id == schema_id
        ]
        prior_support = max(self.schema_support[index] for index in prior_indices)
        prior_matches = max(self.schema_matches[index] for index in prior_indices)
        self.schema_support[latest] = (
            prior_support + predictive_advantage * best_coupling
        )
        self.schema_matches[latest] = prior_matches + 1
        self._update_composition_evidence(latest)
        return {
            "consolidated": True,
            "schema_id": schema_id,
            "coupling": best_coupling,
        }

    def schemas(self, current_anchor: torch.Tensor, capacity: int) -> list[dict]:
        """Return supported transformation schemas in the current tangent space."""
        if capacity <= 0:
            return []
        qualified_ids = {
            schema_id
            for index, schema_id in enumerate(self.schema_ids)
            if self.schema_matches[index] > 0 and self.schema_support[index] > 0.0
        }
        groups: dict[int, list[int]] = {}
        for index, schema_id in enumerate(self.schema_ids):
            if schema_id in qualified_ids:
                groups.setdefault(schema_id, []).append(index)

        records = []
        current_anchor = project_to_sphere_input(current_anchor)
        for schema_id, indices in groups.items():
            representative = max(
                indices,
                key=lambda index: (
                    self.schema_support[index],
                    self.schema_matches[index],
                    index,
                ),
            )
            support = self.schema_support[representative]
            matches = self.schema_matches[representative]
            records.append(
                {
                    "schema_id": schema_id,
                    "support": support,
                    "matches": matches,
                    "strength": support / (1.0 + support),
                    "uncertainty": 1.0 / (1.0 + matches),
                    "action": self.actions[representative],
                    "tangent": parallel_transport_sphere(
                        self.anchors[representative],
                        current_anchor,
                        self.prototypes[representative],
                    ),
                    "evidence": [
                        (
                            self.anchors[index],
                            self.actions[index],
                            self.prototypes[index],
                        )
                        for index in indices
                    ],
                }
            )
        records.sort(key=lambda record: (-record["support"], record["schema_id"]))
        return records[:capacity]

    def composed_schemas(
        self, current_anchor: torch.Tensor, capacity: int
    ) -> list[dict]:
        """Return branches whose added depth earned external predictive evidence."""
        if capacity <= 0:
            return []
        records = self.schemas(
            current_anchor, max(capacity, self.schema_count())
        )
        if not records:
            return []
        by_id = {record["schema_id"]: record for record in records}

        paths = {(record["schema_id"],) for record in records}
        evidence_items = sorted(
            self.composition_observations.items(),
            key=lambda item: (len(item[0]), item[0]),
        )
        for path, observations in evidence_items:
            if observations <= 0 or len(path) > capacity:
                continue
            if any(schema_id not in by_id for schema_id in path):
                continue
            positive = self.composition_positive.get(path, 0.0)
            negative = self.composition_negative.get(path, 0.0)
            prefix = path[:-1]
            prefix_eligible = len(prefix) == 1 or prefix in paths
            if positive > negative and prefix_eligible:
                paths.add(path)

        branches = []
        current_anchor = project_to_sphere_input(current_anchor)
        for path in paths:
            point = current_anchor
            strength = 1.0
            uncertainty_survival = 1.0
            defined = True
            for depth, schema_id in enumerate(path):
                record = by_id[schema_id]
                if depth and bool(antipodal(current_anchor, point).any()):
                    defined = False
                    break
                tangent = parallel_transport_sphere(
                    current_anchor, point, record["tangent"]
                )
                point = exponential_map(point, tangent)
                strength = min(strength, record["strength"])
                uncertainty_survival *= 1.0 - 1.0 / (
                    1.0 + record["matches"]
                )
                if depth:
                    prefix = path[: depth + 1]
                    positive = self.composition_positive[prefix]
                    negative = self.composition_negative.get(prefix, 0.0)
                    observations = self.composition_observations[prefix]
                    strength = min(
                        strength,
                        positive / (1.0 + positive + negative),
                    )
                    uncertainty_survival *= observations / (1.0 + observations)
            if not defined or bool(antipodal(current_anchor, point).any()):
                continue
            first = by_id[path[0]]
            branches.append(
                {
                    "schema_id": path[0],
                    "path": path,
                    "depth": len(path),
                    "support": first["support"],
                    "matches": first["matches"],
                    "strength": strength,
                    "uncertainty": 1.0 - uncertainty_survival,
                    "tangent": log_map_sphere(current_anchor, point),
                    "action": first["action"],
                    "evidence": first["evidence"],
                    "composition_positive": self.composition_positive.get(
                        path, 0.0
                    ),
                    "composition_negative": self.composition_negative.get(
                        path, 0.0
                    ),
                    "composition_observations": self.composition_observations.get(
                        path, 0
                    ),
                }
            )
        branches.sort(
            key=lambda branch: (
                -branch["strength"],
                branch["uncertainty"],
                branch["depth"],
                branch["path"],
            )
        )
        return branches[:capacity]

    def composition_stats(self) -> dict:
        """Detached counts for auditing learned depth without affecting it."""
        eligible_paths: set[tuple[int, ...]] = set()
        ordered_paths = sorted(
            self.composition_observations,
            key=lambda path: (len(path), path),
        )
        for path in ordered_paths:
            positive = self.composition_positive.get(path, 0.0)
            negative = self.composition_negative.get(path, 0.0)
            prefix_supported = len(path) == 2 or path[:-1] in eligible_paths
            if positive > negative and prefix_supported:
                eligible_paths.add(path)
        return {
            "composition_candidates": len(self.composition_observations),
            "composition_eligible": len(eligible_paths),
        }

    def schema_count(self) -> int:
        return len(
            {
                schema_id
                for index, schema_id in enumerate(self.schema_ids)
                if self.schema_matches[index] > 0
                and self.schema_support[index] > 0.0
            }
        )

    def empty(self) -> bool:
        return len(self.prototypes) == 0

    def _compatible_mask(self, action: torch.Tensor) -> torch.Tensor:
        """Boolean mask of prototypes whose action is compatible with ``action``.

        Compatibility is cosine similarity of the released actions; an action
        that is irrelevant (cosine below threshold) cannot lend its deformation
        to a different candidate action.
        """
        action = action.detach().reshape(-1)
        A = torch.stack(self.actions)  # (M, A)
        action = action.to(device=A.device, dtype=A.dtype)
        sim = torch.nn.functional.cosine_similarity(
            A, action.unsqueeze(0), dim=-1
        )
        # Directional compatibility is geometric: non-positive coupling cannot
        # support the queried action. Among positively coupled observations,
        # retain the strongest currently available evidence without a tuned
        # similarity cutoff.
        positive = sim > 0.0
        if not bool(positive.any()):
            return positive
        maximum = sim[positive].max()
        return positive & torch.isclose(sim, maximum, rtol=1e-5, atol=1e-7)

    def residual(self, tangent: torch.Tensor, action: torch.Tensor,
                 current_anchor: torch.Tensor) -> float | None:
        """Normalized orthogonal residual of ``tangent`` w.r.t. the action-compatible
        learned span.

        Prototypes are first parallel-transported to ``current_anchor`` so the
        span is built in a single tangent space, and only action-compatible
        prototypes contribute. Returns None when the buffer is empty or no
        action-compatible prototype exists (unknown reachability).
        """
        t = project_to_tangent(current_anchor.reshape(-1), tangent.detach().reshape(-1))
        if not self.prototypes:
            return None
        mask = self._compatible_mask(action)
        if not bool(mask.any()):
            return None  # incompatible / unknown action coverage
        current_anchor = current_anchor.detach().reshape(-1)
        transported = [
            parallel_transport_sphere(a.detach(), current_anchor, p.detach())
            for a, p in zip(self.anchors, self.prototypes, strict=True)
        ]
        P = torch.stack(transported, dim=0)  # (M, D)
        P = P[mask]
        PPt = P @ P.t()
        Pt = P @ t
        try:
            c = torch.linalg.solve(
                PPt
                + 1e-6
                * torch.eye(PPt.shape[0], device=PPt.device, dtype=PPt.dtype),
                Pt,
            )
        except RuntimeError:
            c = torch.linalg.lstsq(PPt, Pt).solution
        proj = P.t() @ c
        res = t - proj
        norm_res = float(torch.norm(res))
        norm_t = float(torch.norm(t))
        return norm_res / max(norm_t, 1e-8)

    def as_tensor(self) -> torch.Tensor:
        if not self.prototypes:
            return torch.empty(0)
        return torch.stack(self.prototypes, dim=0)


@dataclass
class RecoupleReport:
    per_ghost_error: list[float]
    grounding_updated: bool
    born: bool
    merged: bool
    evicted_count: int
    reachability_prototypes: int
    predictive_advantage: float = 0.0
    schema_consolidated: bool = False
    schema_count: int = 0
    lifecycle_reasons: list[str] = field(default_factory=list)


def recouple(
    population: GhostPopulation,
    real_prev: torch.Tensor,
    observed_anchor: torch.Tensor,
    base_future_anchor: torch.Tensor,
    config: GhostConfig,
    buffer: ReachabilityBuffer,
    released_action: torch.Tensor | None = None,
) -> tuple:
    """Apply one mandatory recoupling and return (new_population, buffer, report)."""
    real_prev = project_to_sphere_input(real_prev)
    observed_anchor = project_to_sphere_input(observed_anchor)
    base_future_anchor = project_to_sphere_input(base_future_anchor)

    # 1. Record the *observed* reachable deformation (external evidence only),
    #    tagged with its source anchor and the released action that produced it.
    obs_def = observed_deformation(real_prev, observed_anchor)
    if released_action is not None:
        buffer.add(real_prev, released_action, obs_def)

    # 2. Contrastive evidence update for each ghost (vs the passive baseline).
    b_err = baseline_error(base_future_anchor, observed_anchor)
    updated = []
    per_ghost_error = []
    for g in population._ghosts:
        action_conditioned = (
            released_action is not None
            and len(g.transfer_actions) >= config.min_action_evidence
            and g.transfer_identifiable(config.min_action_evidence)
        )
        err_tensor = predictive_error(
            g,
            observed_anchor,
            config.transport_step,
            action=released_action if action_conditioned else None,
            anchor=real_prev if action_conditioned else None,
        )
        per_ghost_error.append(float(err_tensor))
        updated.append(
            update_after_recoupling(
                g,
                observed_anchor,
                config.transport_step,
                b_err,
                prediction_error=err_tensor,
            )
        )

    pop_after = GhostPopulation.empty(config.latent_dim, config.effective_capacity)
    pop_after._ghosts = updated

    best_advantage = 0.0
    if per_ghost_error:
        best_advantage = max(0.0, b_err - min(per_ghost_error))
    consolidation = buffer.consolidate_latest(best_advantage)

    # 3. Lifecycle from the observation only (online-statistic rules + diagnostics).
    report = LifecycleReport()
    before_e = len(pop_after)
    pop_after = evict(pop_after, config, report)
    evicted = before_e - len(pop_after)

    before = len(pop_after)
    pop_after = maybe_birth(pop_after, real_prev, observed_anchor, config, report)
    born = len(pop_after) != before

    before_m = len(pop_after)
    pop_after = merge_similar(pop_after, config, report)
    merged = len(pop_after) != before_m

    # 4. Transformation evidence: a newly born ghost owns the deformation
    #    that caused its birth. Otherwise the action-conditioned best predictor
    #    receives the externally observed pair.
    if released_action is not None and len(pop_after._ghosts) > 0:
        if born:
            best = len(pop_after._ghosts) - 1
        else:
            surviving_errors = []
            for ghost in pop_after._ghosts:
                conditioned = ghost.transfer_identifiable(
                    config.min_action_evidence
                )
                surviving_errors.append(
                    float(
                        predictive_error(
                            ghost,
                            observed_anchor,
                            config.transport_step,
                            action=released_action if conditioned else None,
                            anchor=real_prev if conditioned else None,
                        )
                    )
                )
            best = int(torch.tensor(surviving_errors).argmin())
        g = pop_after._ghosts[best]
        pop_after._ghosts[best] = g.add_action_evidence(
            real_prev, released_action, obs_def
        )

    rec_report = RecoupleReport(
        per_ghost_error=per_ghost_error,
        grounding_updated=True,
        born=born,
        merged=merged,
        evicted_count=evicted,
        reachability_prototypes=len(buffer.prototypes),
        predictive_advantage=best_advantage,
        schema_consolidated=consolidation["consolidated"],
        schema_count=buffer.schema_count(),
        lifecycle_reasons=report.reasons,
    )
    return pop_after, buffer, rec_report


def project_to_sphere_input(t):
    from olf.geometry import project_to_sphere
    return project_to_sphere(t.detach().reshape(-1))


def measure_reachability(tangent: torch.Tensor, action: torch.Tensor,
                         current_anchor: torch.Tensor,
                         buffer: ReachabilityBuffer, threshold: float) -> tuple:
    """Return (reachable: bool, residual: float | None).

    Empty buffer or no action-compatible prototype => UNKNOWN => not reachable.
    Otherwise reachable when the normalized residual is within threshold.
    """
    res = buffer.residual(tangent, action, current_anchor)
    if res is None:
        return False, None
    reachable = res <= threshold
    return reachable, res

"""Ghost population integration into the REAL OLF organism.

This module does NOT own a second flow, FLC, transfer, boundary, motor,
semantics, arbitrator, readiness gate, or consequence model. It is constructed
with a reference to the actual ``Organism`` and reads its subsystems, so the
organism remains the single owner of every cognitive operation.

One step (when ``ghost_mode != "off"``):

    begin_step(h)
        -> transport persistent ghosts from the previous real anchor to h
    propose(...)   [only when influences_action]
        -> build the SET of ghost candidates (NO base candidate: the ordinary
           OLF path is the control, never counted as ghost influence)
        -> per candidate: inverse correction (organism FLC transfer),
           action-conditioned prediction (evidence-learned transfer), boundary
           and observational reachability diagnostics
        -> combine corrections into one action delta RELATIVE to the
           ordinary OLF candidate (base_action)
        -> return (delta, diag, influenced, base_future, token)
    <organism releases the action through its OWN motor/boundary/arbitrator>
    recouple_token(token, ...)   [after the environment responds]
        -> organism-owned external recoupling updates each ghost's contrastive
           evidence, reachability buffer, and lifecycle; stores the transformation
           evidence binding the released action to the observed consequence.

When ``ghost_mode == "off"`` the integration object is never created, so the
organism is byte-identical to the pre-integration loop. When ``observe`` the
ghost state receives detached post-recoupling evidence but returns a zero
action delta (no influence, no gradient, no RNG/training-order change).
"""

import torch

from olf.geometry import (
    angular_distance,
    antipodal,
    exponential_map,
    log_map_sphere,
    parallel_transport_sphere,
    project_to_sphere,
    project_to_tangent,
)

from .config import GhostConfig
from .diagnostics import (
    assert_no_prohibited_labels,
    check_sphere_norm,
    check_tangent_validity,
    finite_or_raise,
)
from .population import GhostPopulation
from .recoupling import ReachabilityBuffer, measure_reachability, recouple
from .trajectory import make_ghost


class GhostIntegration:
    """Owns the ghost population but delegates every cognitive operation to the
    organism's actual subsystems."""

    def __init__(self, config: GhostConfig, organism):
        if not isinstance(config, GhostConfig):
            raise TypeError("config must be a GhostConfig")
        self.config = config
        self.organism = organism

        # Private RNG (never the global generator) for random routing only.
        self._rng = torch.Generator()
        if config.seed is not None:
            self._rng.manual_seed(config.seed)

        if config.effective_capacity == 0:
            self.population = GhostPopulation.empty(config.latent_dim, 0)
        else:
            # Start EMPTY: no invented axis, no ghost until external evidence.
            self.population = GhostPopulation(config.latent_dim, config.effective_capacity)
        self.buffer = ReachabilityBuffer(config.reachability_capacity)
        self.prev_anchor = None
        self._recoupled_since_influence = True
        self._pending_token = None
        # Atomic pending-release transaction: the token, the
        # real latent before release, the passive baseline, the ghost candidates
        # that actually influenced, and (filled at recouple) the released action.
        self._pending_transaction = None
        self._aborted_at_reset = False
        self._births_total = 0
        self._merges_total = 0
        self._evictions_total = 0
        self._recouplings_total = 0
        self._last_tension = None

        # Hard audit: the ghost config / population must not carry prohibited labels.
        assert_no_prohibited_labels(config)
        self._audit_population()

    # ---- audit -----------------------------------------------------------
    def _audit_population(self) -> None:
        for g in self.population._ghosts:
            assert_no_prohibited_labels(g._as_dict())

    def telemetry(self) -> dict:
        """Return detached diagnostics without changing ghost state."""
        ghosts = self.population._ghosts
        count = len(ghosts)
        if count:
            evidence = sum(float(g.evidence_support) for g in ghosts) / count
            grounding = sum(float(g.grounding) for g in ghosts) / count
            transfer_support = sum(len(g.transfer_actions) for g in ghosts)
        else:
            evidence = 0.0
            grounding = 0.0
            transfer_support = 0
        branches = (
            (
                self.buffer.composed_schemas(
                    self.prev_anchor, self.config.effective_capacity
                )
                if self.config.schema_composition_enabled
                else self.buffer.schemas(
                    self.prev_anchor, self.config.effective_capacity
                )
            )
            if self.prev_anchor is not None
            else []
        )
        composition_stats = self.buffer.composition_stats()
        return {
            "births_total": self._births_total,
            "merges_total": self._merges_total,
            "evictions_total": self._evictions_total,
            "recouplings_total": self._recouplings_total,
            "population": count,
            "evidence_support_mean": evidence,
            "grounding_mean": grounding,
            "transfer_support": transfer_support,
            "schemas": self.buffer.schema_count(),
            "composed_branches": sum(
                branch.get("depth", 1) > 1 for branch in branches
            ),
            "max_schema_depth": max(
                (branch.get("depth", 1) for branch in branches), default=0
            ),
            **composition_stats,
        }

    def tension(self) -> dict:
        """Return current grounded future disagreement without changing state."""
        with torch.no_grad():
            return self.population.tension(self.config.transport_step)

    def _record_recoupling(self, report) -> None:
        self._recouplings_total += 1
        self._births_total += int(report.born)
        self._merges_total += int(report.merged)
        self._evictions_total += int(report.evicted_count)

    # ---- reset (episodic ghost state) -----------------------------------
    def reset(self) -> None:
        """Clear pending release and the (temporary) ghost population.

        The learned reachable-deformation space (self.buffer) persists across
        episodes: it is the organism's own deformation vocabulary, not a
        temporary trajectory. Aborting a pending transaction is recorded.
        """
        aborted_pending = self._pending_token is not None
        self.population = GhostPopulation(
            self.config.latent_dim, self.config.effective_capacity
        )
        self.buffer.start_segment()
        self.prev_anchor = None
        self._recoupled_since_influence = True
        self._pending_token = None
        self._pending_transaction = None
        self._aborted_at_reset = aborted_pending
        self._last_tension = None

    # ---- step lifecycle --------------------------------------------------
    def begin_step(self, real_anchor: torch.Tensor) -> None:
        """Transport persistent ghosts to the current real anchor."""
        h = project_to_sphere(real_anchor.detach().reshape(1, -1))[0]
        self._audit_population()
        if len(self.population) == 0:
            if self.config.schema_composition_enabled:
                schemas = self.buffer.composed_schemas(
                    h, self.config.effective_capacity
                )
            else:
                schemas = self.buffer.schemas(
                    h, self.config.effective_capacity
                )
            if schemas:
                restored = GhostPopulation.empty(
                    self.config.latent_dim, self.config.effective_capacity
                )
                for schema in schemas:
                    ghost = make_ghost(
                        anchor=h,
                        tangent=schema["tangent"],
                        credibility=schema["strength"],
                        grounding=schema["strength"],
                        uncertainty=schema["uncertainty"],
                        evidence_support=schema["support"],
                        horizon_expr=1.0 / max(float(schema.get("depth", 1)), 1.0),
                    )
                    for source, action, tangent in schema["evidence"]:
                        ghost = ghost.add_action_evidence(
                            source, action, tangent
                        )
                    restored.append(ghost)
                self.population = restored
            self.prev_anchor = h.detach().reshape(-1)
            return
        if self.prev_anchor is None or not self.config.persistence_enabled:
            # No history or persistence disabled: re-anchor each ghost at the
            # current real point (its tangent is parallel-transported so it
            # stays a valid tangent, but no deformation is invented).
            new_ghosts = []
            for g in self.population._ghosts:
                transp = parallel_transport_sphere(g.anchor, h, g.tangent)
                new_ghosts.append(
                    g.with_updates(anchor=h, tangent=project_to_tangent(h, transp))
                )
            pop = GhostPopulation.empty(self.config.latent_dim, self.config.effective_capacity)
            pop._ghosts = new_ghosts
            self.population = pop
        else:
            prev = project_to_sphere(self.prev_anchor.reshape(1, -1))[0]
            with torch.no_grad():
                self.population = self.population.transport(
                    prev, h, self.config.transport_step
                )
                for g in self.population._ghosts:
                    assert check_tangent_validity(g.anchor, g.tangent)
        self.prev_anchor = h.detach().reshape(-1)

    def propose(self, sigma_flat, self_state, sigma_t, base_action):
        """Return (delta, diag, influenced, base_future, token).

        ``delta`` is a (1, A) detached tensor to ADD to the organism's candidate
        action. In ``off``/``observe`` it is zero. The organism then runs its own
        boundary + motor release on the modified candidate, so mode/readiness/
        verdict/viability are all the organism's real ones.
        """
        h = project_to_sphere(self.organism.h.detach().reshape(1, -1))[0]
        base_action = base_action.detach().reshape(1, -1)
        sigma_flat = sigma_flat.detach().reshape(1, -1)
        self_state = self_state.detach().reshape(1, -1)

        # Passive baseline: the latent deformation the ORDINARY OLF
        # candidate action is predicted to produce, via the organism's own
        # consequence machinery. This is the control prediction, NOT a desired
        # future latent, and is never counted as ghost influence.
        base_future = self._passive_baseline_anchor(h, sigma_t, base_action)
        zero = torch.zeros_like(base_action)

        diag = {
            "ghost_mode": self.config.ghost_mode,
            "ablation": self.config.ablation,
            "population": len(self.population),
            "base_future": base_future,
        }
        self._last_tension = self.tension()
        diag["tension_before"] = self._last_tension

        if not self.config.influences_action:
            diag["influenced"] = False
            return zero, diag, False, base_future, None

        # Gate: an influenced release requires a recoupling since the last one.
        if not self._recoupled_since_influence:
            diag["influenced"] = False
            diag["waiting_for_recoupling"] = True
            return zero, diag, False, base_future, None

        # no_recoupling blocks influence entirely (token never issued).
        if not self.config.recoupling_enabled:
            diag["influenced"] = False
            diag["blocked_no_recoupling"] = True
            return zero, diag, False, base_future, None

        delta, diag, influenced, ghost_indices = self._influence(
            h, sigma_flat, self_state, base_action, diag, sigma_t=sigma_t
        )

        if influenced:
            # Issue an opaque pending-release token consumed exactly once by the
            # post-environment recoupling hook, and store the atomic transaction.
            token = object()
            self._pending_token = token
            self._recoupled_since_influence = False
            self._pending_transaction = {
                "token": token,
                "real_prev": self.organism.h.detach().clone(),
                "base_future_anchor": base_future.detach().clone(),
                "ghost_indices": ghost_indices,
                "released_action": None,
                "finalized": False,
                "tension_before": self._last_tension,
            }
            return delta, diag, True, base_future, token
        return zero, diag, False, base_future, None

    @property
    def has_pending_release(self) -> bool:
        return self._pending_token is not None

    def finalize_release(
        self,
        token,
        *,
        real_prev: torch.Tensor,
        released_action,
    ) -> None:
        """Atomically bind a proposal token to the actual motor release."""
        if token is None or token is not self._pending_token:
            raise ValueError("cannot finalize a missing or foreign ghost release")
        transaction = self._pending_transaction
        if transaction is None or transaction["token"] is not token:
            raise RuntimeError("ghost release transaction is inconsistent")
        if transaction["finalized"]:
            raise RuntimeError("ghost release transaction was already finalized")
        transaction["real_prev"] = project_to_sphere(
            real_prev.detach().reshape(-1)
        ).clone()
        transaction["released_action"] = _as_action_tensor(
            released_action, like=transaction["real_prev"]
        ).clone()
        transaction["finalized"] = True

    def pending_context(self, token):
        """Return frozen release context only for the matching live token."""
        if token is None or token is not self._pending_token:
            return None
        transaction = self._pending_transaction
        if transaction is None or not transaction["finalized"]:
            raise RuntimeError("pending ghost release has no finalized motor action")
        return transaction

    def _passive_baseline_anchor(self, h, sigma_t, base_action):
        """Predict the consequence of the ORDINARY OLF candidate action.

        Uses the organism's own consequence model on the ordinary candidate
        action ``base_action``; the resulting tangent deformation is exponentiated
        to a future latent. No desired-future latent is used as the baseline.
        """
        with torch.no_grad():
            consequences = self.organism.semantics.predict_consequences(
                sigma_t.detach(), base_action.detach()
            )
            affordance = self.organism._compute_entity_affordance(consequences)
            weights = torch.softmax(affordance, dim=-1)
            predicted_effect = (weights.unsqueeze(-1) * consequences["dh_pred"]).sum(dim=1)
            tangent = project_to_tangent(h, predicted_effect[0])
            return exponential_map(h, tangent)

    def _influence(
        self, h, sigma_flat, self_state, base_action, diag, sigma_t=None
    ):
        """Combine only ghost candidates; the ordinary OLF path is the control.

        Returns (delta, diag, influenced, ghost_indices) where delta is the
        ghost correction RELATIVE to the ordinary OLF candidate ``base_action``.
        """
        if self.config.centroid_before_inverse:
            return self._centroid_before_inverse(
                h, sigma_flat, self_state, base_action, diag
            )

        ghost_idx = list(range(len(self.population)))
        weights = h.new_zeros(len(ghost_idx))
        credibility_mass = 0.0
        grounded_credibility_mass = 0.0
        combined_action = torch.zeros_like(base_action[0])
        reachable_flags = []
        ghost_indices = []
        valid_candidates = []
        step = self.config.transport_step

        for local_i, g_index in enumerate(ghost_idx):
            g = self.population._ghosts[g_index]
            # --- detached cognitive ops (ghost memory stays detached) --------
            with torch.no_grad():
                # Signature target (fixed observed deformation).
                target = g.predicted_anchor(step)
                if bool(antipodal(h, target).any()):
                    reachable_flags.append(False)
                    diag.setdefault("candidates", []).append({
                        "ghost_index": g_index,
                        "action_conditioned": False,
                        "action_support_known": False,
                        "reachable": False,
                        "reachability_residual": None,
                        "motor_valid": False,
                        "inverse_defined": False,
                        "undefined_reason": "antipodal_target",
                    })
                    continue
                correction = self.organism.flc.transfer.inverse_correction(
                    h.unsqueeze(0), target.unsqueeze(0)
                )[0]
                correction = project_to_tangent(h, correction)
                proj_in = torch.cat(
                    [h, target, correction, sigma_flat[0], self_state[0], base_action[0]],
                    dim=-1,
                )
                cand_action0 = self.organism.flc.motor_projection(proj_in.unsqueeze(0))[0]

                action_conditioned = (
                    len(g.transfer_actions) >= self.config.min_action_evidence
                    and g.transfer_identifiable(self.config.min_action_evidence)
                )
                # Once grounded evidence defines an action-deformation map,
                # invert that map directly for the ghost's desired tangent.
                # Before then the candidate remains diagnostic and cannot act.
                if action_conditioned:
                    desired_tangent = parallel_transport_sphere(
                        g.anchor, h, g.tangent
                    ) * g.horizon_expr
                    cand_action = torch.clamp(
                        g.transfer_inverse(desired_tangent, h), -1.0, 1.0
                    )
                    pred_tan = g.transfer_predict(cand_action, h)
                    target2 = exponential_map(h, pred_tan * step)
                    if bool(antipodal(h, target2).any()):
                        reachable_flags.append(False)
                        diag.setdefault("candidates", []).append({
                            "ghost_index": g_index,
                            "action_conditioned": True,
                            "action_support_known": True,
                            "reachable": False,
                            "reachability_residual": None,
                            "motor_valid": False,
                            "inverse_defined": False,
                            "undefined_reason": "antipodal_action_target",
                        })
                        continue
                else:
                    cand_action = cand_action0
                motor_valid = bool((cand_action.abs() <= 1.0 + 1e-3).all())

            # Reachability remains observational; it does not suppress a ghost.
            res = None
            if not self.config.reachability_enabled:
                reachable = True
            else:
                gtan = parallel_transport_sphere(g.anchor, h, g.tangent)
                reachable, res = measure_reachability(
                    gtan, cand_action, h, self.buffer, self.config.reachability_threshold
                )

            reachable_flags.append(reachable)
            candidate_diag = {
                "ghost_index": g_index,
                "action_conditioned": action_conditioned,
                "action_support_known": res is not None,
                "reachable": reachable,
                "reachability_residual": res,
                "motor_valid": motor_valid,
                "inverse_defined": True,
            }
            diag.setdefault("candidates", []).append(candidate_diag)

            if self.config.random_routing:
                w = float(torch.rand((), generator=self._rng)) + 1e-8
            else:
                credibility = max(float(g.credibility), 0.0)
                grounding = max(float(g.grounding), 0.0)
                externally_grounded = grounding > 0.0
                if action_conditioned and externally_grounded:
                    credibility_mass += credibility
                    grounded_credibility_mass += credibility * grounding
                    w = credibility
                else:
                    w = 0.0
                candidate_diag["predictive_credibility"] = credibility
                candidate_diag["externally_grounded"] = externally_grounded
            weights[local_i] = w
            combined_action = combined_action + w * cand_action
            ghost_indices.append(g_index)
            valid_candidates.append(
                {
                    "ghost_index": g_index,
                    "local_index": local_i,
                    "action": cand_action,
                    "diag": candidate_diag,
                }
            )

        total = float(weights.sum())
        diag["reachable_count"] = int(sum(reachable_flags))
        diag["weights"] = [float(w) for w in weights]
        diag["routing"] = (
            "random" if self.config.random_routing else "predictive_credibility"
        )
        diag["motor_valid_fraction"] = float(motor_valid_count(diag))
        diag["ghost_count_influencing"] = len(ghost_indices)

        if total <= 0.0:
            diag["influenced"] = False
            diag["no_candidates"] = True
            return torch.zeros_like(base_action[0]), diag, False, ghost_indices

        combined_action = combined_action / total
        combined_action = torch.clamp(combined_action, -1.0, 1.0)
        support_amplitude = (
            1.0
            if self.config.random_routing
            else min(
                1.0,
                grounded_credibility_mass / max(credibility_mass, 1e-8),
            )
        )
        synthesis = None
        if not self.config.random_routing and sigma_t is not None:
            with torch.no_grad():
                synthesis = self._synthesize(
                    h=h,
                    sigma_flat=sigma_flat,
                    self_state=self_state,
                    sigma_t=sigma_t,
                    base_action=base_action[0],
                    support_amplitude=support_amplitude,
                    weights=weights,
                    candidates=valid_candidates,
                )
        if synthesis is not None:
            diag.update(synthesis["diag"])
            if synthesis["selected_ordinary"]:
                diag["influenced"] = False
                return (
                    torch.zeros_like(base_action[0]),
                    diag,
                    False,
                    ghost_indices,
                )
            combined_action = synthesis["action"]
        # Delta relative to the ordinary OLF candidate (the unmodified path).
        delta = (
            combined_action - base_action[0]
            if synthesis is not None
            else support_amplitude * (combined_action - base_action[0])
        ).reshape(1, -1)
        diag["influenced"] = True
        diag["support_amplitude"] = float(support_amplitude)
        return delta, diag, True, ghost_indices

    def _synthesize(
        self,
        *,
        h,
        sigma_flat,
        self_state,
        sigma_t,
        base_action,
        support_amplitude,
        weights,
        candidates,
    ):
        """Cross-evaluate actions under every grounded transformation model.

        No scalar synthesis reward is introduced. Candidate futures are compared
        on three non-dominated quantities: model disagreement, mismatch to OLF's
        current future latent, and existing boundary-attributable danger.
        """
        model_indices = [
            index
            for index, ghost in enumerate(self.population._ghosts)
            if float(weights[index]) > 0.0
            and ghost.transfer_identifiable(self.config.min_action_evidence)
        ]
        supported_candidates = [
            candidate
            for candidate in candidates
            if float(weights[candidate["local_index"]]) > 0.0
        ]
        if not model_indices or not supported_candidates:
            return None

        model_weights = weights[model_indices]
        model_weights = model_weights / model_weights.sum()
        desired_future = self.organism.flc.future_field(
            h.unsqueeze(0), sigma_flat, self_state
        ).latent[0]

        zero_action = torch.zeros_like(base_action).unsqueeze(0)
        zero_effect = self._predicted_effect(sigma_t, zero_action)
        zero_risk = self.organism.veto.predict_risk(
            h.unsqueeze(0), zero_action, zero_effect.unsqueeze(0)
        )[0, 0]

        action_rows = [
            {
                "action": base_action,
                "source": None,
                "support": 0.0,
                "diag": None,
            }
        ]
        action_rows.extend(
            {
                "action": torch.clamp(
                    base_action
                    + support_amplitude * (candidate["action"] - base_action),
                    -1.0,
                    1.0,
                ),
                "source": candidate["ghost_index"],
                "support": float(weights[candidate["local_index"]]),
                "diag": candidate["diag"],
            }
            for candidate in supported_candidates
        )

        objectives = []
        valid_rows = []
        for row in action_rows:
            action = row["action"]
            predicted_futures = []
            for model_index in model_indices:
                ghost = self.population._ghosts[model_index]
                predicted = ghost.transfer_predict(action, h)
                predicted_futures.append(
                    exponential_map(h, predicted * self.config.transport_step)
                )
            futures = torch.stack(predicted_futures)
            pairwise = angular_distance(
                futures[:, None, :], futures[None, :, :]
            )
            pair_weights = model_weights[:, None] * model_weights[None, :]
            disagreement = 0.5 * torch.sum(
                pair_weights * pairwise.square()
            )
            resultant = torch.sum(model_weights[:, None] * futures, dim=0)
            if float(resultant.norm()) <= torch.finfo(resultant.dtype).eps:
                continue
            consensus = project_to_sphere(resultant)
            future_mismatch = angular_distance(consensus, desired_future)

            predicted_effect = self._predicted_effect(
                sigma_t, action.unsqueeze(0)
            )
            risk = self.organism.veto.predict_risk(
                h.unsqueeze(0),
                action.unsqueeze(0),
                predicted_effect.unsqueeze(0),
            )[0, 0]
            danger = torch.relu(risk - zero_risk)
            metric = (
                float(danger),
                float(disagreement),
                float(future_mismatch),
            )
            objectives.append(metric)
            valid_rows.append(row)
            if row["diag"] is not None:
                row["diag"].update(
                    {
                        "synthesis_danger": metric[0],
                        "synthesis_disagreement": metric[1],
                        "synthesis_future_mismatch": metric[2],
                    }
                )

        if len(valid_rows) < 2:
            return None
        front = _non_dominated_indices(objectives)
        ordinary_index = next(
            (
                index
                for index, row in enumerate(valid_rows)
                if row["source"] is None
            ),
            None,
        )
        if ordinary_index is None:
            return None
        ordinary_objective = objectives[ordinary_index]
        dominating_ghosts = [
            index
            for index, row in enumerate(valid_rows)
            if row["source"] is not None
            and all(
                left <= right
                for left, right in zip(
                    objectives[index], ordinary_objective, strict=True
                )
            )
            and any(
                left < right
                for left, right in zip(
                    objectives[index], ordinary_objective, strict=True
                )
            )
        ]
        selected = (
            max(
                dominating_ghosts,
                key=lambda index: (valid_rows[index]["support"], -index),
            )
            if dominating_ghosts
            else ordinary_index
        )
        selected_row = valid_rows[selected]
        selected_source = selected_row["source"]
        return {
            "action": torch.clamp(selected_row["action"], -1.0, 1.0),
            "selected_ordinary": selected_source is None,
            "diag": {
                "routing": "pareto_synthesis",
                "synthesis_model_count": len(model_indices),
                "synthesis_candidate_count": len(valid_rows),
                "synthesis_pareto_indices": front,
                "synthesis_dominating_ghosts": dominating_ghosts,
                "synthesis_selected_ghost": selected_source,
                "synthesis_objectives": objectives,
            },
        }

    def _predicted_effect(self, sigma_t, action):
        consequences = self.organism.semantics.predict_consequences(
            sigma_t.detach(), action.detach()
        )
        affordance = self.organism._compute_entity_affordance(consequences)
        entity_weights = torch.softmax(affordance, dim=-1)
        return (
            entity_weights.unsqueeze(-1) * consequences["dh_pred"]
        ).sum(dim=1)[0]

    def _centroid_before_inverse(
        self,
        h,
        sigma_flat,
        self_state,
        base_action,
        diag,
    ):
        """Ablation: collapse future points before one inverse-transfer query."""
        future_points = []
        future_weights = []
        source_indices = []
        step = self.config.transport_step
        with torch.no_grad():
            for index, ghost in enumerate(self.population._ghosts):
                if ghost.transfer_identifiable(self.config.min_action_evidence):
                    predicted = ghost.transfer_predict(base_action[0], h)
                    future = exponential_map(h, predicted * step)
                else:
                    future = ghost.predicted_anchor(step)
                future_points.append(future)
                future_weights.append(max(float(ghost.credibility), 1e-8))
                source_indices.append(index)

            if not future_points or sum(future_weights) <= 0.0:
                diag.update(
                    {
                        "influenced": False,
                        "no_candidates": True,
                        "centroid_before_inverse": True,
                        "ghost_count_influencing": 0,
                        "weights": future_weights,
                    }
                )
                return torch.zeros_like(base_action[0]), diag, False, []

            weighted_sum = sum(
                weight * point
                for weight, point in zip(future_weights, future_points, strict=True)
            )
            if float(weighted_sum.norm()) <= torch.finfo(weighted_sum.dtype).eps:
                diag.update(
                    {
                        "influenced": False,
                        "no_candidates": True,
                        "centroid_before_inverse": True,
                        "centroid_undefined": True,
                        "ghost_count_influencing": 0,
                        "weights": future_weights,
                    }
                )
                return torch.zeros_like(base_action[0]), diag, False, []
            centroid = project_to_sphere(weighted_sum)
            correction = self.organism.flc.transfer.inverse_correction(
                h.unsqueeze(0), centroid.unsqueeze(0)
            )[0]
            correction = project_to_tangent(h, correction)
            projection_input = torch.cat(
                [h, centroid, correction, sigma_flat[0], self_state[0], base_action[0]],
                dim=-1,
            )
            candidate = self.organism.flc.motor_projection(
                projection_input.unsqueeze(0)
            )[0]
            motor_valid = bool((candidate.abs() <= 1.0 + 1e-3).all())

        residual = None
        if self.config.reachability_enabled:
            deformation = log_map_sphere(h, centroid)
            reachable, residual = measure_reachability(
                deformation,
                candidate,
                h,
                self.buffer,
                self.config.reachability_threshold,
            )
        else:
            reachable = True

        diag.update(
            {
                "centroid_before_inverse": True,
                "reachable_count": int(reachable),
                "weights": future_weights,
                "routing": "future_centroid",
                "motor_valid_fraction": float(motor_valid),
                "candidates": [
                    {
                        "ghost_indices": source_indices,
                        "action_conditioned": True,
                        "action_support_known": residual is not None,
                        "reachable": reachable,
                        "reachability_residual": residual,
                        "motor_valid": motor_valid,
                    }
                ],
            }
        )
        combined = torch.clamp(candidate, -1.0, 1.0)
        diag["influenced"] = True
        diag["ghost_count_influencing"] = len(source_indices)
        return (
            (combined - base_action[0]).reshape(1, -1),
            diag,
            True,
            source_indices,
        )

    # ---- external recoupling (token-enforced) ---------------------------
    def recouple_token(self, token, real_prev: torch.Tensor,
                       observed_anchor: torch.Tensor,
                       base_future_anchor: torch.Tensor,
                       released_action=None) -> dict:
        """Mandatory, token-enforced recoupling after the environment responds.

        A non-None ``token`` must equal the pending token issued by propose and
        must not have been consumed; otherwise this is not a genuine external
        recoupling and raises an error. A None token (observe mode, or a step where no
        influence occurred) simply updates evidence when recoupling is enabled.
        A None token MUST NOT bypass a pending transaction.
        """
        if not self.config.active:
            return {"updated": False, "population": len(self.population)}

        # a None token cannot bypass a live pending release.
        if self._pending_token is not None and token is None:
            raise ValueError("missing token cannot bypass a pending ghost release")

        if token is not None:
            if token is not self._pending_token:
                raise ValueError("recoupling token does not match pending release")
            transaction = self.pending_context(token)
            if transaction is None:
                raise RuntimeError("pending ghost release context is missing")
            if not self.config.recoupling_enabled:
                self._recoupled_since_influence = False
                return {"updated": False, "blocked_no_recoupling": True,
                        "population": len(self.population)}
            frozen_prev = transaction["real_prev"]
            frozen_base = transaction["base_future_anchor"]
            frozen_action = transaction["released_action"]
            tension_before = transaction["tension_before"]
            pop_after, self.buffer, report = recouple(
                self.population, frozen_prev, observed_anchor,
                frozen_base, self.config, self.buffer,
                released_action=frozen_action,
            )
            self.population = pop_after
            self._audit_population()
            self._recoupled_since_influence = True
            self._pending_token = None
            self.prev_anchor = project_to_sphere(
                observed_anchor.detach().reshape(1, -1)
            )[0].detach().reshape(-1)
            self._pending_transaction = None
            self._record_recoupling(report)
            return self._recoupling_result(report, tension_before)

        # No token: observe mode or a non-influenced step.
        if not self.config.recoupling_enabled:
            self._recoupled_since_influence = False
            return {"updated": False, "blocked_no_recoupling": True,
                    "population": len(self.population)}
        ra = _as_action_tensor(released_action, like=real_prev)
        pop_after, self.buffer, report = recouple(
            self.population, real_prev, observed_anchor,
            base_future_anchor, self.config, self.buffer,
            released_action=ra,
        )
        self.population = pop_after
        self._audit_population()
        self._recoupled_since_influence = True
        self.prev_anchor = project_to_sphere(
            observed_anchor.detach().reshape(1, -1)
        )[0].detach().reshape(-1)
        self._record_recoupling(report)
        return self._recoupling_result(report, self._last_tension)

    def _recoupling_result(self, report, tension_before):
        tension_after = self.tension()
        tension_reduction = None
        if (
            tension_before is not None
            and tension_before.get("defined", False)
            and tension_after.get("defined", False)
        ):
            tension_reduction = max(
                0.0,
                tension_before["normalized"] - tension_after["normalized"],
            )
        return {
            "updated": True,
            "population": len(self.population),
            "born": report.born,
            "merged": report.merged,
            "evicted": report.evicted_count,
            "reachability_prototypes": report.reachability_prototypes,
            "per_ghost_error": report.per_ghost_error,
            "lifecycle_reasons": report.lifecycle_reasons,
            "predictive_advantage": report.predictive_advantage,
            "schema_consolidated": report.schema_consolidated,
            "schema_count": report.schema_count,
            "tension_before": tension_before,
            "tension_after": tension_after,
            "tension_reduction": tension_reduction,
        }

    # ---- finite / invariant guards --------------------------------------
    def check_invariants(self) -> None:
        if len(self.population) == 0:
            return
        anchors = self.population.anchors()
        finite_or_raise("ghost_anchors", anchors)
        assert check_sphere_norm(anchors), "ghost anchors left the sphere"
        for i in range(len(self.population)):
            assert check_tangent_validity(self.population[i].anchor,
                                          self.population[i].tangent), \
                f"ghost {i} tangent invalid"


def _as_action_tensor(released_action, like=None):
    if released_action is None:
        return None
    device = None if like is None else like.device
    dtype = torch.float32 if like is None else like.dtype
    return torch.as_tensor(
        released_action, dtype=dtype, device=device
    ).reshape(-1)


def motor_valid_count(diag):
    cands = diag.get("candidates", [])
    if not cands:
        return 1.0
    return sum(1 for c in cands if c.get("motor_valid", True)) / len(cands)


def _non_dominated_indices(objectives):
    """Return indices not strictly dominated on all minimized objectives."""
    front = []
    for index, current in enumerate(objectives):
        dominated = False
        for other_index, other in enumerate(objectives):
            if index == other_index:
                continue
            no_worse = all(left <= right for left, right in zip(other, current, strict=True))
            strictly_better = any(
                left < right for left, right in zip(other, current, strict=True)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(index)
    return front

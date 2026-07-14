"""Persistent lifecycle for active, dormant, and free transfer ghosts."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .set_control import BranchBelief
from .transfer_ghosts import (
    TransferGhostField,
    _directions,
    _errors,
    _target_directions,
    fit_transfer_ghosts,
)


@dataclass
class GhostSlot:
    identity: int
    matrix: torch.Tensor
    weight: torch.Tensor
    probe_arc: torch.Tensor
    endpoint_arc: torch.Tensor
    scale: torch.Tensor
    active: bool
    last_supported: int
    support_windows: int


@dataclass(frozen=True)
class PopulationEvent:
    phase: int
    kind: str
    identity: int


class PersistentTransferGhostField:
    """Bounded transfer memory with evidence-based dormancy and reactivation."""

    def __init__(self, latent_dim: int, capacity: int = 8) -> None:
        self.latent_dim = latent_dim
        self.capacity = capacity
        self.phase = -1
        self.next_identity = 0
        self.slots: list[GhostSlot] = []
        self.events: list[PopulationEvent] = []

    def _distance(
        self, left: torch.Tensor, right: torch.Tensor, current: torch.Tensor
    ) -> float:
        left_direction = _directions(current, left.unsqueeze(0))[:, 0]
        right_direction = _directions(current, right.unsqueeze(0))[:, 0]
        angle = torch.acos(
            (left_direction * right_direction).sum(dim=-1).clamp(-1.0, 1.0)
        )
        return float(angle.mean().item())

    def _match(
        self, provisional: TransferGhostField, current: torch.Tensor
    ) -> dict[int, int]:
        candidates = []
        for provisional_index, matrix in enumerate(provisional.matrices):
            for slot_index, slot in enumerate(self.slots):
                distance = self._distance(matrix, slot.matrix, current)
                evidence_radius = float(
                    3.0 * (provisional.scale[provisional_index] + slot.scale).item()
                )
                if distance <= evidence_radius:
                    candidates.append((distance, provisional_index, slot_index))
        matches = {}
        used_slots = set()
        for _, provisional_index, slot_index in sorted(candidates):
            if provisional_index not in matches and slot_index not in used_slots:
                matches[provisional_index] = slot_index
                used_slots.add(slot_index)
        return matches

    def _allocate(self, provisional: TransferGhostField, index: int) -> int:
        if len(self.slots) >= self.capacity:
            dormant = [
                (slot.last_supported, position)
                for position, slot in enumerate(self.slots)
                if not slot.active
            ]
            if not dormant:
                raise RuntimeError("ghost capacity exhausted with no dormant slot")
            _, position = min(dormant)
            evicted = self.slots.pop(position)
            self.events.append(PopulationEvent(self.phase, "evict", evicted.identity))
        identity = self.next_identity
        self.next_identity += 1
        self.slots.append(
            GhostSlot(
                identity=identity,
                matrix=provisional.matrices[index].clone(),
                weight=provisional.weights[index].clone(),
                probe_arc=provisional.probe_arc[index].clone(),
                endpoint_arc=provisional.endpoint_arc[index].clone(),
                scale=provisional.scale[index].clone(),
                active=True,
                last_supported=self.phase,
                support_windows=1,
            )
        )
        self.events.append(PopulationEvent(self.phase, "birth", identity))
        return len(self.slots) - 1

    def _known_support(
        self, current: torch.Tensor, endpoints: torch.Tensor
    ) -> set[int]:
        """Recognize stored laws directly, including rare dormant laws."""
        if not self.slots:
            return set()
        matrices = torch.stack([slot.matrix for slot in self.slots])
        targets = _target_directions(current, endpoints)
        errors = _errors(_directions(current, matrices), targets)
        assignment = errors.argmin(dim=1)
        supported = set()
        for index, slot in enumerate(self.slots):
            selected = assignment == index
            if int(selected.sum().item()) < self.latent_dim:
                continue
            radius = float(3.0 * slot.scale.item())
            if float(errors[selected, index].mean().item()) <= radius:
                supported.add(index)
        return supported

    def update(
        self,
        contexts: torch.Tensor,
        probes: torch.Tensor,
        endpoints: torch.Tensor,
        *,
        seed: int,
    ) -> None:
        self.phase += 1
        provisional = fit_transfer_ghosts(
            contexts, probes, endpoints, capacity=self.capacity, seed=seed
        )
        current = contexts[:, : self.latent_dim]
        matches = self._match(provisional, current)
        supported_slots = set(matches.values()) | self._known_support(
            current, endpoints
        )
        for slot_index in supported_slots:
            slot = self.slots[slot_index]
            if not slot.active:
                slot.active = True
                self.events.append(
                    PopulationEvent(self.phase, "reactivate", slot.identity)
                )
            slot.last_supported = self.phase
        for slot_index, slot in enumerate(self.slots):
            if slot_index not in supported_slots and slot.active:
                slot.active = False
                self.events.append(
                    PopulationEvent(self.phase, "dormant", slot.identity)
                )
        for provisional_index in range(len(provisional.matrices)):
            if provisional_index not in matches:
                matches[provisional_index] = self._allocate(
                    provisional, provisional_index
                )
                continue
            slot = self.slots[matches[provisional_index]]
            if not slot.active:
                slot.active = True
                self.events.append(
                    PopulationEvent(self.phase, "reactivate", slot.identity)
                )
            slot.weight = provisional.weights[provisional_index].clone()
            slot.probe_arc = provisional.probe_arc[provisional_index].clone()
            slot.endpoint_arc = provisional.endpoint_arc[provisional_index].clone()
            slot.last_supported = self.phase
            slot.support_windows += 1

    def belief(self, context: torch.Tensor) -> BranchBelief:
        active = [slot for slot in self.slots if slot.active]
        if not active:
            raise RuntimeError("no active ghosts")
        field = TransferGhostField(
            matrices=torch.stack([slot.matrix for slot in active]),
            weights=torch.stack([slot.weight for slot in active]),
            probe_arc=torch.stack([slot.probe_arc for slot in active]),
            endpoint_arc=torch.stack([slot.endpoint_arc for slot in active]),
            scale=torch.stack([slot.scale for slot in active]),
            events=(),
        )
        weights = field.weights / field.weights.sum()
        field = TransferGhostField(
            field.matrices,
            weights,
            field.probe_arc,
            field.endpoint_arc,
            field.scale,
            (),
        )
        return field.belief(context)

    def active_identities(self) -> tuple[int, ...]:
        return tuple(slot.identity for slot in self.slots if slot.active)

    def dormant_identities(self) -> tuple[int, ...]:
        return tuple(slot.identity for slot in self.slots if not slot.active)

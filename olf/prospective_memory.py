"""Content-addressed episodic memory for prospective event transformations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProspectiveEventMemory(nn.Module):
    """Store situated causes before later organism-native events.

    Records keep raw observation ingredients so keys can be re-encoded by the
    current semantic binder at read time. Entity indices identify a source
    instance only inside the stored observation; retrieval itself is entirely
    content-addressed and permutation equivariant.
    """

    def __init__(
        self,
        obs_dim,
        latent_dim,
        action_dim,
        max_records=1024,
    ):
        super().__init__()
        self.max_records = int(max_records)
        self.register_buffer(
            "observations", torch.zeros(max_records, obs_dim)
        )
        self.register_buffer(
            "spm_traces", torch.zeros(max_records, latent_dim)
        )
        self.register_buffer(
            "entity_indices", torch.zeros(max_records, dtype=torch.long)
        )
        self.register_buffer(
            "endpoints", torch.zeros(max_records, latent_dim)
        )
        self.register_buffer(
            "future_values", torch.zeros(max_records, 1)
        )
        self.register_buffer("risks", torch.zeros(max_records, 1))
        self.register_buffer(
            "actions", torch.zeros(max_records, action_dim)
        )
        self.register_buffer("horizons", torch.zeros(max_records, 1))
        self.register_buffer("write_index", torch.zeros((), dtype=torch.long))
        self.register_buffer("record_count", torch.zeros((), dtype=torch.long))

    def __len__(self):
        return int(self.record_count.item())

    def add(
        self,
        *,
        observation,
        spm_trace,
        entity_index,
        endpoint,
        future_value,
        risk,
        action,
        horizon,
    ):
        index = int(self.write_index.item())
        with torch.no_grad():
            self.observations[index].copy_(
                torch.as_tensor(
                    observation,
                    dtype=torch.float32,
                    device=self.observations.device,
                ).reshape_as(self.observations[index])
            )
            self.spm_traces[index].copy_(
                torch.as_tensor(
                    spm_trace,
                    dtype=torch.float32,
                    device=self.spm_traces.device,
                ).reshape_as(self.spm_traces[index])
            )
            self.entity_indices[index] = int(entity_index)
            self.endpoints[index].copy_(
                F.normalize(
                    torch.as_tensor(
                        endpoint,
                        dtype=torch.float32,
                        device=self.endpoints.device,
                    ).reshape_as(self.endpoints[index]),
                    p=2,
                    dim=-1,
                )
            )
            self.future_values[index, 0] = float(future_value)
            self.risks[index, 0] = float(risk)
            self.actions[index].copy_(
                torch.as_tensor(
                    action,
                    dtype=torch.float32,
                    device=self.actions.device,
                ).reshape_as(self.actions[index])
            )
            self.horizons[index, 0] = float(horizon)
            self.write_index.fill_((index + 1) % self.max_records)
            self.record_count.fill_(min(self.max_records, len(self) + 1))

    def records(self):
        count = len(self)
        if count == 0:
            return None
        return {
            "observations": self.observations[:count],
            "spm_traces": self.spm_traces[:count],
            "entity_indices": self.entity_indices[:count],
            "endpoints": self.endpoints[:count],
            "future_values": self.future_values[:count],
            "risks": self.risks[:count],
            "actions": self.actions[:count],
            "horizons": self.horizons[:count],
        }

    def read(self, query_sigma, memory_keys, top_k=8):
        records = self.records()
        if records is None:
            return None
        if query_sigma.ndim != 3 or memory_keys.ndim != 2:
            raise ValueError("expected query (B,N,H) and keys (M,H)")
        count = memory_keys.shape[0]
        k = min(int(top_k), count)
        query = F.normalize(query_sigma, p=2, dim=-1)
        keys = F.normalize(memory_keys, p=2, dim=-1)
        similarity = torch.einsum("bnh,mh->bnm", query, keys)
        top_similarity, top_indices = similarity.topk(k, dim=-1)
        temperature = 1.0 / math.sqrt(float(query_sigma.shape[-1]))
        weights = torch.softmax(top_similarity / temperature, dim=-1)

        def gather(values):
            flat_indices = top_indices.reshape(-1)
            selected = values[flat_indices].reshape(
                *top_indices.shape, *values.shape[1:]
            )
            return (weights.unsqueeze(-1) * selected).sum(dim=-2)

        endpoint = F.normalize(gather(records["endpoints"]), p=2, dim=-1)
        max_similarity = top_similarity[..., 0].clamp(-1.0, 1.0)
        support = torch.exp(
            -(1.0 - max_similarity)
            * math.sqrt(float(query_sigma.shape[-1]))
        )
        maturity = count / float(count + k)
        support = support * maturity
        return {
            "future_latent": endpoint,
            "value": gather(records["future_values"]),
            "risk": gather(records["risks"]),
            "action": gather(records["actions"]),
            "horizon": gather(records["horizons"]),
            "support": support,
            "max_similarity": max_similarity,
        }

    def clear(self):
        with torch.no_grad():
            self.write_index.zero_()
            self.record_count.zero_()

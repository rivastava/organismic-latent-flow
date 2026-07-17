"""Seed-0 engineering assay for ghost integration in the real OLF Organism.

This is not a benchmark and makes no performance claim. It exercises the
production ownership chain against an existing environment observation stream:

    Organism.select_action -> motor release -> environment transition
    -> Organism.learn_consequence -> token-enforced ghost recoupling

The ghost subsystem receives only the real latent before release, the released
action, the ordinary consequence-model baseline, and the observed next latent.
Environment reward and completion metadata are not passed to ghost recoupling.
"""

import sys

import numpy as np
import torch

from benchmarks.self_state_meaning import SelfStateMeaningEnv
from olf.ghosts.config import GhostConfig
from olf.organism import Organism
from olf.seeding import set_seed


def _organism(config: GhostConfig) -> Organism:
    return Organism(
        obs_dim=18,
        latent_dim=config.latent_dim,
        hidden_dim=64,
        ghost_mode=config.ghost_mode,
        ghost_config=config,
    )


def run_ghost_assay(
    config: GhostConfig,
    steps: int = 24,
    seed: int = 0,
) -> dict:
    """Run a bounded integration assay without training or hidden signals."""
    set_seed(seed)
    config = GhostConfig(**{**config.__dict__, "seed": seed})
    organism = _organism(config)
    organism.eval()
    env = SelfStateMeaningEnv(seed=seed)
    obs = env.reset()

    history = []
    for step in range(steps):
        action, info = organism.select_action(obs, evaluate=True)
        if not np.isfinite(action).all() or not torch.isfinite(organism.h).all():
            raise ValueError(f"non-finite organism state at step {step}")

        next_obs, _environment_signal, done, _metadata = env.step(action)
        token_before = organism._ghost_token
        organism.learn_consequence(
            0.0,
            0.0,
            0.0,
            0.0,
            next_obs=next_obs,
        )
        if token_before is not None and organism._ghost_token is not None:
            raise RuntimeError("external recoupling did not consume the release token")

        ghost_diag = info.get("ghost") or {}
        history.append(
            {
                "step": step,
                "mode": config.ghost_mode,
                "ablation": config.ablation,
                "population": 0 if organism.ghost is None else len(organism.ghost.population),
                "influenced": bool(ghost_diag.get("ghost_influenced", False)),
                "reachable": ghost_diag.get("reachable_count"),
                "verdict": info["verdict"],
                "token_consumed": token_before is None or organism._ghost_token is None,
            }
        )
        obs = env.reset() if done else next_obs
        if done:
            organism.reset_state()

    return {
        "mode": config.ghost_mode,
        "ablation": config.ablation,
        "steps": steps,
        "final_population": 0 if organism.ghost is None else len(organism.ghost.population),
        "reachability_prototypes": (
            0 if organism.ghost is None else len(organism.ghost.buffer.prototypes)
        ),
        "history": history,
    }


def main() -> int:
    modes = [
        GhostConfig(ghost_mode="off"),
        GhostConfig(ghost_mode="observe"),
        GhostConfig(ghost_mode="influence"),
        GhostConfig(ghost_mode="influence", ablation="no_ghosts"),
        GhostConfig(ghost_mode="influence", ablation="single_ghost"),
        GhostConfig(ghost_mode="influence", ablation="no_persistence"),
        GhostConfig(ghost_mode="influence", ablation="centroid_before_inverse"),
        GhostConfig(ghost_mode="influence", ablation="no_recoupling"),
        GhostConfig(ghost_mode="influence", ablation="no_reachability"),
        GhostConfig(ghost_mode="influence", ablation="random_routing"),
    ]
    print("Ghost integration engineering assay (seed 0 only)")
    for config in modes:
        result = run_ghost_assay(config, steps=24, seed=0)
        influenced = sum(row["influenced"] for row in result["history"])
        print(
            f"  [{config.ghost_mode:9s}] abl={str(config.ablation):23s} "
            f"population={result['final_population']} "
            f"prototypes={result['reachability_prototypes']} "
            f"influenced={influenced}"
        )
    print("assay complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())

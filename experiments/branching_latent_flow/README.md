# Branching Latent Flow

This experiment studies whether OLF can preserve several possible latent
continuations without abandoning its shared spherical substrate.

The working principle is:

> A latent has no permanent semantic type. Its role comes from its relation to
> the organism's present flow, observed consequence, and current use.

The experiment is isolated from the OLF organism. It does not modify FLC,
boundary risk, memory, arbitration, motor control, or benchmark behavior.

## Architecture

```text
completed observed trajectories
-> evidence-supported transfer laws
-> several continuations on S^(d-1)
-> one tangent correction per continuation
-> observed-flow recoupling
-> situated consequence deformation
```

The learner receives completed observable trajectory tensors. World
annotations are confined to synthetic scheduling and offline measurement.
Transfer discovery does not receive continuation count, branch index, reward,
success, task label, or hidden generator state.

The population has fixed maximum capacity but begins with one transfer law.
A new law is retained only when paired held-out evidence and a description
length criterion support the additional capacity. Changing-world experiments
allow unsupported laws to become dormant and later return.

Each retained transfer can be materialized as a temporary spherical trajectory.
The same points can then be read relationally as a possible future, a present
correction, a remembered continuation, or a predecessor. These reads do not
change grounding. Only an external observation can recouple credibility and
grounding.

Situated consequence traces contain:

```text
before, deformation, after, self_state,
viability, viability_delta, boundary_deformation
```

They contain no functional-role target. Consequence changes temporary
influence, persistence, uncertainty, and boundary risk while preserving the
underlying trajectory.

## Verified Results

The following values come from deterministic ten-seed synthetic experiments
with hidden continuation counts of 2, 3, and 5.

| Study | Observed result |
| --- | --- |
| Transfer discovery | Correct population count in all 30 runs |
| Future coverage | 99.98% to 100% |
| Branch action plausibility | 98% or higher |
| Centroid action plausibility | 0.4% to 0.9% |
| Changing-world return | Same stored transfer returned in 10/10 runs |
| Persistent phase coverage | 100% |
| Role-free future and action reads | 100% coverage and plausibility |
| Observation recoupling | 100% selected continuation recovery |
| Situated consequence selection | 97.43% to 98.67% |
| Fixed-context consequence control | 23.23% to 49.93% |
| Scrambled-consequence control | 18.50% to 51.67% |

The consequence study also exposes the present research boundary:
removing the independent boundary gate produced the same result as the full
condition. In this synthetic world, immediate desirability and safety align.
Independent boundary necessity therefore requires a delayed conflict protocol
where an attractive short-horizon continuation can become lethal later.

These measurements establish properties of the isolated synthetic mechanism.
Closed-loop influence on the OLF organism begins only after the passive
integration and leakage gates are complete.

## Layout

- `protocol.py`: held-out spherical world and offline annotations.
- `observations.py`: observable-only training interface.
- `set_control.py`: set-valued futures, inverse corrections, and recoupling.
- `transfer_ghosts.py`: evidence-supported transfer discovery.
- `persistent_ghosts.py`: bounded active/dormant lifecycle scaffolding.
- `role_free_ghosts.py`: shared spherical trajectories and relational reads.
- `consequence_deformation.py`: situated consequence learning.
- `*_experiment.py`: deterministic reproductions that write to `results/`.

## Run

Focused tests:

```bash
pytest tests/test_branching_*.py
```

Ten-seed reproductions:

```bash
python -m experiments.branching_latent_flow.transfer_experiment
python -m experiments.branching_latent_flow.lifecycle_experiment
python -m experiments.branching_latent_flow.trajectory_experiment
python -m experiments.branching_latent_flow.consequence_experiment
```

Outputs are written under gitignored `results/`. No generated result is part
of the source tree.

## Current Boundary

The next gate introduces variable horizons and a delayed
desirability-versus-danger conflict. It must demonstrate that an independent
boundary process changes selection while keeping continuation identity hidden
from the learner. OLF integration follows in two steps: passive observation
first, then closed-loop influence through FLC, boundary, readiness, motor
projection, and mandatory recoupling.

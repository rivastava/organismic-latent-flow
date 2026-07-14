# Organismic Latent Flow

Organismic Latent Flow (OLF) is a research prototype for studying an
organism-like agent whose internal state evolves as a continuous latent flow.

The project explores whether memory, action, boundary risk, prospective
control, and recoupling can be represented as roles of one continuous
organismic process.

> **Experimental branch:** `experimental/branching-latent-flow` studies a
> bounded population of interchangeable spherical continuations: transfer-law
> discovery, persistence, relational reuse, recoupling, and situated
> consequence deformation. It remains isolated from the OLF organism. See
> [`experiments/branching_latent_flow/`](experiments/branching_latent_flow/).

## Core Idea

OLF keeps a latent state `h(t)` on the unit sphere. The organism observes a
simple environment, updates its latent flow, binds the current situation,
forms action pressure, checks boundary risk, releases motor action, and then
recouples through observed consequence.

The intended loop is:

```text
observation
-> current latent flow
-> situated binding
-> prospective event memory
-> future-latent control
-> boundary check
-> motor projection
-> consequence
-> recoupling update
```

## Future-Latent Control

Future-Latent Control (FLC) is a subsystem inside OLF. It is not the whole
architecture.

Current implementation:

```text
current latent
-> remembered or generated future latent
-> inverse transfer correction
-> action projection
-> boundary check
-> motor release
```

The FLC code lives in:

- `olf/future.py`
- `olf/transfer.py`
- `olf/boundary.py`
- `olf/organism.py`

Observed entity transitions can ground a future endpoint and endogenous
viability value over a delayed horizon. FLC projects the selected endpoint
backward through an inverse transfer field to produce present corrective
pressure. This event-grounded path is an explicit research configuration;
the base organism remains available for controlled ablation.

## What Is Implemented

- Continuous latent flow on a unit sphere.
- Spherical phase memory and RTCM-style causal memory.
- Situated consequence binding with self-state/context/object inputs.
- Future-latent control as an explicit OLF subsystem.
- Bounded event-based prospective memory with delayed endogenous credit.
- Grounded inverse action transfer from future endpoints.
- Boundary-risk network `B_psi(h, a, dh_pred)`.
- Motor release with release/hold/recouple/rollback verdicts.
- Basic benchmark environments.
- Smoke tests and focused diagnostic runners.
- Deterministic seed handling for experiments.

## Research Scope

OLF is currently focused on small organismic control settings where the agent
must bind self-state, context, memory, and action under survival pressure.
The active research boundary is transfer: the organism can identify a
value-bearing future in the current benchmarks, but motor correction does not
yet generalize reliably across randomized geometry or procedural access
changes.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Test

```bash
python tests/smoke_test.py
pytest
python -m py_compile $(git ls-files "*.py")
```

Optional lint/type commands:

```bash
ruff check .
mypy olf experiments benchmarks tests
```

## Run Experiments

Core suite:

```bash
python -m experiments.run_core
```

Focused boundary/FLC diagnostic:

```bash
python -m experiments.boundary_signal
```

Prospective-memory ablation and paired analysis:

```bash
python -m experiments.diagnostics.probe_prospective_ablation \
  --seeds 10 --workers 6 \
  --conditions babble_only full \
  --tasks self_state_meaning delayed_lure triadic_binding target_threat \
    abstraction_unseen_randompos \
  --out results/diagnostics/prospective_core_10seed.json

python -m experiments.diagnostics.analyze_paired_results \
  results/diagnostics/prospective_core_10seed.json
```

Generated outputs are ignored by git under `results/` and
`experiments/learning_curves/`.

## License

MIT. See `LICENSE`.

## Current Research Status

The current prospective configuration uses only terminal body viability and
lethal collapse as its training consequence. An end-to-end reward-scrambling
test changes benchmark reward from `+10,000` to `-10,000` while keeping the
trajectory fixed and requires every learned tensor and memory buffer to remain
bit-identical.

With 10 paired seeds, 150 training episodes, 15 evaluation episodes, and the
same policy-independent motor-babbling control in both conditions:

| Task | Babbling control | Prospective OLF | Difference |
| --- | ---: | ---: | ---: |
| Self-state meaning | 36.0% | 72.0% | +36.0 pp |
| Delayed lure | 30.0% | 90.0% | +60.0 pp |
| Triadic binding | 25.3% | 56.7% | +31.3 pp |
| Target threat | 30.0% | 60.0% | +30.0 pp |
| Randomized-position abstraction | 8.7% | 14.0% | +5.3 pp |

The first three tasks have positive paired bootstrap intervals and unadjusted
exact sign-flip tests below `0.04`. No task remains significant after Holm
correction across all five comparisons at this sample size. Randomized-position
performance remains near floor. These results establish a functional
prospective-memory contribution in the fixed-layout tasks while leaving broad
geometric generalization unresolved.

Boundary risk is trained as action-attributable change in proximity to the
organism's viability boundary. This avoids the earlier all-zero death-event
target and keeps need pressure separate from irreversible-risk estimation.

## Repository Layout

```text
olf/
  events.py
  geometry.py
  flow.py
  future.py
  prospective.py
  prospective_memory.py
  transfer.py
  boundary.py
  motor.py
  organism.py
benchmarks/
experiments/
  diagnostics/
tests/
```

The public repository intentionally keeps the research narrative in this
README. Old stage reports, generated results, caches, and exploratory scratch
files are not part of the publishable tree.

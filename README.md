# Organismic Latent Flow

Organismic Latent Flow (OLF) is a research prototype for studying an
organism-like agent whose internal state evolves as a continuous latent flow.

The project explores whether memory, action, boundary risk, prospective
control, and recoupling can be represented as roles of one continuous
organismic process.

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
-> future latent
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

The current FLC implementation exposes the structural path used by OLF:
task-dependent future latents are projected backward through an inverse
transfer field to produce present corrective pressure.

## What Is Implemented

- Continuous latent flow on a unit sphere.
- Spherical phase memory and RTCM-style causal memory.
- Situated consequence binding with self-state/context/object inputs.
- Future-latent control as an explicit OLF subsystem.
- Boundary-risk network `B_psi(h, a, dh_pred)`.
- Motor release with release/hold/recouple/rollback verdicts.
- Basic benchmark environments.
- Smoke tests and focused diagnostic runners.
- Deterministic seed handling for experiments.

## Research Scope

OLF is currently focused on small organismic control settings where the agent
must bind self-state, context, memory, and action under survival pressure. The
main open engineering questions are:

- Measuring FLC contribution through controlled ablations.
- Calibrating boundary risk under rare and gradual failure modes.
- Extending horizon control beyond short benchmark episodes.
- Stabilizing benchmark variance across seeds.
- Turning the current research modules into stronger reusable learning
  components.

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

Generated outputs are ignored by git under `results/` and
`experiments/learning_curves/`.

## License

MIT. See `LICENSE`.

## Current Research Status

The present checkpoint is v0.3.2.11. The important change is that boundary risk
is no longer trained only from rare death events; it now receives a
self-supervised body-boundary proximity signal.

This addresses the previous all-zero `B_psi` failure mode at the signal-design
level. Current work is focused on benchmark stability, FLC ablations, and
boundary-risk calibration.

## Repository Layout

```text
olf/
  geometry.py
  flow.py
  future.py
  transfer.py
  boundary.py
  motor.py
  memory.py
  organism.py
benchmarks/
experiments/
tests/
```

The public repository intentionally keeps the research narrative in this
README. Old stage reports, generated results, caches, and exploratory scratch
files are not part of the publishable tree.

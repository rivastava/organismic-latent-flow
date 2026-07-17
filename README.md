# Organismic Latent Flow

Organismic Latent Flow (OLF) is a research implementation of an organism-like
control loop built around a continuous latent state on the unit sphere.

The implementation treats memory, prospective control, boundary assessment,
motor release, and consequence recoupling as interacting parts of one control
process. Future-Latent Control (FLC) is a subsystem of OLF: a future latent is
formed at a task-dependent horizon, mapped through inverse transfer into a
present correction, checked against boundary risk, and released through the
motor system.

```text
observation
-> latent flow
-> situated binding and memory
-> future latent
-> inverse-transfer correction
-> boundary check
-> motor release
-> observed consequence
-> recoupling
```

## Included

- Continuous latent flow on the unit sphere.
- Spherical phase memory and rotary temporal-causal memory.
- Situated consequence binding over entity, context, and self-state inputs.
- Future-Latent Control and inverse transfer.
- Consequence memory and prospective event memory.
- Action-attributable boundary-risk assessment.
- Motor release with release, hold, recouple, and rollback verdicts.
- Role-free ghost trajectories for bounded alternative continuations.
- Synthetic control environments, deterministic experiment runners, and tests.

Ghost trajectories remain optional and default to `off`. When enabled, they are
born from observed latent deformation, accumulate action-consequence evidence,
are checked by the existing boundary and motor systems, and are recoupled after
the environment responds. They do not introduce a separate policy, reward
channel, or privileged environment state.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Verify

```bash
ruff check .
python tests/smoke_test.py
pytest
```

## Run

Core benchmark suite:

```bash
python -m experiments.run_core
```

Boundary and FLC diagnostic:

```bash
python -m experiments.boundary_signal
```

Ghost integration assay:

```bash
python -m experiments.ghost_integration
```

Generated outputs are written below `results/` and are excluded from version
control.

## Layout

```text
olf/           Core OLF implementation
benchmarks/    Synthetic control environments
experiments/   Reproducible runners and diagnostics
tests/         Unit, integration, and smoke tests
```

## Scope

OLF is experimental research software. The included environments provide
mechanism-specific evidence for control, memory, boundary, and transfer.

## License

MIT. See `LICENSE`.

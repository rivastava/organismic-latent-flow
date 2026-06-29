"""Compact trajectory visualization for OLF organisms.

Runs one deterministic task/seed, trains briefly, then executes a single
evaluation episode and saves a 2x2 diagnostic figure. Signals are
observational only: no architecture changes, no threshold manipulation.

Usage:
    python experiments/visualize_trajectory.py
    python -c "from experiments.visualize_trajectory import run_trajectory; run_trajectory()"

Output:
    results/figures/trajectory.png (gitignored)
"""

import os

import numpy as np


def run_trajectory(task="self_state_meaning", seed=0, out_path=None, train_episodes=100):
    """Train briefly, run one eval episode, save a 2x2 trajectory figure.

    Parameters
    ----------
    task : str
        Benchmark task name (must be a key in experiments.run_core.ENV_MAP).
    seed : int
        Deterministic seed for Python, NumPy, and PyTorch.
    out_path : str or None
        Where to write the PNG. Defaults to results/figures/trajectory.png.
    train_episodes : int
        Number of training episodes before the diagnostic rollout. The default
        is intentionally short; tests can lower it further.

    Returns
    -------
    str
        Absolute path to the saved figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from olf.organism import Organism
    from experiments.run_core import ENV_MAP, train_agent, set_global_seed

    set_global_seed(seed)

    if task not in ENV_MAP:
        known = ", ".join(sorted(ENV_MAP))
        raise ValueError(f"Unknown task '{task}'. Known tasks: {known}")

    if out_path is None:
        out_path = os.path.join("results", "figures", "trajectory.png")

    # --- Train briefly (smoke-quality, not benchmark-quality) ---
    agent = Organism(obs_dim=18, action_dim=3)
    agent = train_agent(agent, task, num_episodes=train_episodes, lr=0.01, seed=seed)
    danger_threshold = getattr(getattr(agent, "veto", None), "threshold", None)

    # --- Single evaluation episode, collecting per-step signals ---
    env = ENV_MAP[task](seed=seed)
    obs = env.reset()
    agent.eval()
    agent.episode_count = 9999  # past warmup
    agent.reset_state()

    agent_positions = []
    entity_positions = []
    dangers = []
    verdicts = []
    action_norms = []
    status = "timeout"

    done = False
    while not done:
        agent_pos = obs[0:2].copy()
        agent_positions.append(agent_pos)

        # Entity world-space positions from relative obs
        ent0_pos = agent_pos + obs[6:8]
        ent1_pos = agent_pos + obs[14:16]
        entity_positions.append((ent0_pos.copy(), ent1_pos.copy()))

        action, action_info = agent.select_action(obs, evaluate=True)
        next_obs, reward, done, env_info = env.step(action)

        dangers.append(float(action_info.get("danger", 0.0)))
        verdicts.append(action_info.get("verdict", "release"))
        action_norms.append(float(np.linalg.norm(action)))

        was_lethal = 1.0 if env_info["status"] in ("death", "starvation") else 0.0
        agent.learn_consequence(
            reward,
            was_lethal,
            next_obs[2] - obs[2],
            next_obs[3] - obs[3],
        )

        status = env_info["status"]
        obs = next_obs

    agent_positions = np.array(agent_positions)
    steps = len(dangers)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(
        f"OLF trajectory: {task}  seed={seed}  status={status}  steps={steps}",
        fontsize=11,
        fontweight="bold",
    )

    # (0,0) Agent path + entity positions
    ax = axes[0, 0]
    ax.plot(
        agent_positions[:, 0],
        agent_positions[:, 1],
        "-o",
        color="#2563eb",
        linewidth=1.2,
        markersize=2.5,
        label="agent path",
        zorder=2,
    )
    ax.plot(agent_positions[0, 0], agent_positions[0, 1], "s", color="#16a34a",
            markersize=7, label="start", zorder=3)
    ax.plot(agent_positions[-1, 0], agent_positions[-1, 1], "D", color="#dc2626",
            markersize=7, label="end", zorder=3)
    if entity_positions:
        e0 = np.array([p[0] for p in entity_positions])
        e1 = np.array([p[1] for p in entity_positions])
        ax.plot(e0[:, 0], e0[:, 1], "s", color="#f59e0b", markersize=9,
                label="entity 0", zorder=3)
        ax.plot(e1[:, 0], e1[:, 1], "^", color="#7c3aed", markersize=9,
                label="entity 1", zorder=3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7, loc="best")
    ax.set_title("Agent path + entities")

    # (0,1) Danger over time
    ax = axes[0, 1]
    ax.fill_between(range(steps), dangers, alpha=0.3, color="#dc2626")
    ax.plot(dangers, color="#dc2626", linewidth=1.0)
    if danger_threshold is not None:
        ax.axhline(
            float(danger_threshold),
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label="release threshold",
        )
    ax.set_xlim(0, max(steps - 1, 1))
    ax.set_xlabel("step")
    ax.set_ylabel("danger")
    ax.set_title("Boundary danger")
    ax.legend(fontsize=7)

    # (1,0) Verdict over time
    ax = axes[1, 0]
    verdict_colors = {
        "release": "#16a34a",
        "recouple": "#f59e0b",
        "rollback": "#dc2626",
        "hold": "#6b7280",
    }
    for i, v in enumerate(verdicts):
        ax.axvspan(i, i + 1, color=verdict_colors.get(v, "#d1d5db"), alpha=0.85)
    # Legend (one patch per unique verdict)
    seen = set()
    for v in verdicts:
        if v not in seen:
            ax.plot([], [], color=verdict_colors.get(v, "#d1d5db"), linewidth=6,
                    label=v)
            seen.add(v)
    ax.set_xlim(0, max(steps - 1, 1))
    ax.set_yticks([])
    ax.set_xlabel("step")
    ax.set_title("Veto verdict")
    ax.legend(fontsize=7, loc="best")

    # (1,1) Action norm over time
    ax = axes[1, 1]
    ax.plot(action_norms, color="#0891b2", linewidth=1.0)
    ax.set_xlim(0, max(steps - 1, 1))
    ax.set_xlabel("step")
    ax.set_ylabel("||action||")
    ax.set_title("Action norm")

    plt.tight_layout()
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    return os.path.abspath(out_path)


if __name__ == "__main__":
    path = run_trajectory()
    print(f"Saved: {path}")

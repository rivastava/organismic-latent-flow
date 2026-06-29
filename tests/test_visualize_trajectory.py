"""Tests for experiments/visualize_trajectory.py."""

import os
import tempfile

from experiments.visualize_trajectory import run_trajectory


def test_run_trajectory_returns_path():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "traj.png")
        result = run_trajectory(
            task="target_threat", seed=0, out_path=out, train_episodes=3
        )
        assert isinstance(result, str)
        assert result.endswith("traj.png")


def test_run_trajectory_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "traj.png")
        run_trajectory(task="target_threat", seed=0, out_path=out, train_episodes=3)
        assert os.path.isfile(out)
        size = os.path.getsize(out)
        assert size > 1000, f"Figure too small ({size} bytes), likely empty"


def test_run_trajectory_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        out1 = os.path.join(tmp, "a.png")
        out2 = os.path.join(tmp, "b.png")
        run_trajectory(task="target_threat", seed=42, out_path=out1, train_episodes=3)
        run_trajectory(task="target_threat", seed=42, out_path=out2, train_episodes=3)
        assert os.path.getsize(out1) == os.path.getsize(out2)


def test_run_trajectory_different_tasks():
    with tempfile.TemporaryDirectory() as tmp:
        for task in ["target_threat", "self_state_meaning"]:
            out = os.path.join(tmp, f"{task}.png")
            run_trajectory(task=task, seed=0, out_path=out, train_episodes=3)
            assert os.path.isfile(out)

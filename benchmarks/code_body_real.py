"""A compact code-repair environment with mandatory recoupling.

The organism lives in a tiny "broken program" world:
  - The program is a list of integers (1D bytecode) with a known target.
  - Body actions (all from [-1, 1], thresholded at 0.5 in the env):
      inspect_file  : read current value at index = round((u+1)/2 * (n-1))
                      and observe; does not mutate the program.
      run_test      : compute hamming distance to target; observe result;
                      does not mutate but consumes a recouple step.
      edit_code     : change the value at the current cursor to
                      round((u+1)/2 * 9); mutates the program. MUST be
                      followed by a recouple phase (no edits in next 2 steps).
      rollback      : revert the program to its pre-edit snapshot.
      hold          : no movement; satisfies the recouple requirement.

The organism must learn the inspect → edit → recouple → test cycle.

The three-dimensional continuous action ``(dx, dy, u)`` maps to a discrete
environmental effect:
  u > 0.5  → action = edit_code (with strength)
  -0.5 < u < 0.5 → inspect
  u < -0.5 → rollback
  recoupling is enforced if last step was edit: action must be hold
  (which means low dx, dy, neutral u).

After an edit, the next step must be a hold. Otherwise the program crashes.

This is a real-enough code body to test recoupling, invention (when no
edit works), and goal-as-attractor (viable programs).
"""

import numpy as np
from benchmarks.base import BaseBenchmark


class CodeBodyRealEnv(BaseBenchmark):
    """Real-ish code repair: edit a 1D integer program to match a target.

    Action mapping (3-dim continuous):
      dx, dy  : movement; ignored inside code body (no spatial nav)
      u       : action intent
                u >  0.5  → edit_code with strength (u + 1) / 2 * 9
                0.5 < u < 0.5  → inspect_file
                u < -0.5  → rollback
    """

    def __init__(self, seed=None, program_len=4, max_steps=40):
        super().__init__(seed=seed)
        self.max_steps = max_steps
        self.program_len = program_len
        self.target_program = None
        self.current_program = None
        self.last_action_was_edit = False
        self.recouple_remaining = 0
        self.edit_history = []
        self.snapshot = None

    def setup_task(self):
        # The "code body": a 1D program of integers in [0, 9].
        # The target is a different random program.
        # Position layout: still 2D for the obs interface, but dx/dy are
        # used as a "cursor" for inspection/edit.
        self.target_program = self.rng.integers(0, 10, size=self.program_len).astype(np.int32)
        self.current_program = self.rng.integers(0, 10, size=self.program_len).astype(np.int32)
        # Ensure they differ so the task is non-trivial.
        while np.array_equal(self.target_program, self.current_program):
            self.current_program = self.rng.integers(0, 10, size=self.program_len).astype(np.int32)
        # Snapshot for rollback.
        self.snapshot = self.current_program.copy()
        # Entity 0 = the "code block" target.
        # Features: 4-dim embedding of (cursor_pos_norm, edit_count, last_diff, success_signal).
        self.entities[0]["pos"] = np.array([0.0, 0.0], dtype=np.float32)
        self.entities[0]["feats"] = np.array(
            [0.0, 0.0, self._program_diff(), 0.0], dtype=np.float32
        )
        # Entity 1 = the "test runner" / inspection target.
        self.entities[1]["pos"] = np.array([0.5, 0.0], dtype=np.float32)
        self.entities[1]["feats"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # Context: "code" mode.
        self.context = np.array([0.0, 1.0], dtype=np.float32)
        # Cursor state (in self_state for the organism to track).
        self.self_state[0] = 0.0  # cursor position (normalized 0..1)
        self.self_state[1] = 0.0  # recoupling timer
        self.last_action_was_edit = False
        self.recouple_remaining = 0
        self.edit_history = []
        self.snapshot = self.current_program.copy()
        # Track actions for diagnostics.
        self._n_edits = 0
        self._n_inspects = 0
        self._n_rollbacks = 0
        self._n_crashes = 0
        self._n_recouples = 0

    def _program_diff(self):
        """Hamming distance / program length (in [0, 1])."""
        return float(np.mean(self.current_program != self.target_program))

    def _action_intent(self, u):
        """Map continuous u to discrete intent."""
        if u > 0.5:
            return "edit_code", float((u + 1.0) / 2.0)  # strength in [0.5, 1.0]
        elif u < -0.5:
            return "rollback", float((u + 1.0) / 2.0)
        else:
            return "inspect", float((u + 1.0) / 2.0)

    def step(self, action):
        dx, dy, u, done, info = self.base_step(action)
        # dx, dy are ignored in the code body but recorded for the
        # organism's "motor" awareness.
        intent, strength = self._action_intent(u)

        # An edit must be followed by a stationary recoupling step.
        if self.last_action_was_edit and self.recouple_remaining > 0:
            # The next step must be a hold (no movement, no edit).
            is_moving = (abs(dx) > 0.01 or abs(dy) > 0.01)
            is_edit_again = (intent == "edit_code")
            if is_moving or is_edit_again:
                # Violation → system crash.
                self.viability = 0.0
                done = True
                info["status"] = "death"
                self._n_crashes += 1
            else:
                # Successfully recoupled.
                self._n_recouples += 1
            self.recouple_remaining -= 1

        # Apply the intent.
        if not done and intent == "edit_code":
            # Edit at the cursor position.
            cursor_idx = int(round(self.self_state[0] * (self.program_len - 1)))
            cursor_idx = max(0, min(self.program_len - 1, cursor_idx))
            new_val = int(round(strength * 9.0))
            self.snapshot = self.current_program.copy()
            old_val = int(self.current_program[cursor_idx])
            self.current_program[cursor_idx] = new_val
            self.edit_history.append((cursor_idx, old_val, new_val))
            self.last_action_was_edit = True
            self.recouple_remaining = 2  # must hold for next 2 steps
            self._n_edits += 1
        elif not done and intent == "rollback":
            if self.snapshot is not None:
                self.current_program = self.snapshot.copy()
            self.last_action_was_edit = False
            self.recouple_remaining = 0
            self._n_rollbacks += 1
        else:
            # Inspect.
            self.last_action_was_edit = False
            self.recouple_remaining = 0
            self._n_inspects += 1

        # Update cursor (dx, dy move the cursor in [0, 1]).
        if not done:
            self.self_state[0] = float(np.clip(self.self_state[0] + 0.1 * dx, 0.0, 1.0))
            # self_state[1] tracks how stable the system is (resets on good recouple).
            if not self.last_action_was_edit:
                self.self_state[1] = max(0.0, self.self_state[1] - 0.05)

        # Update entity features to reflect program state.
        if not done:
            self.entities[0]["feats"] = np.array(
                [self.self_state[0], float(self._n_edits) / 10.0,
                 self._program_diff(), float(len(self.edit_history)) / 10.0],
                dtype=np.float32,
            )
            self.entities[1]["feats"] = np.array(
                [float(self._n_inspects) / 10.0,
                 float(self._n_rollbacks) / 10.0,
                 float(self._n_crashes) / 10.0,
                 float(self._n_recouples) / 10.0],
                dtype=np.float32,
            )

        # Emit reward only when the program reaches the target state.
        reward = 0.0
        if self._program_diff() < 0.01:
            # Program matches target.
            self.entities[0]["active"] = False
            done = True
            info["status"] = "success"
            reward = 1.0
            self.self_state[0] = 0.0

        # Starvation check (system drift).
        if not done:
            self.self_state[1] = float(np.clip(self.self_state[1] + 0.02, 0.0, 1.0))
            if self.self_state[1] >= 1.0:
                self.viability = 0.0
                done = True
                info["status"] = "starvation"

        # Time limit.
        self.step_count += 1
        if not done and self.step_count >= self.max_steps:
            done = True
            if info["status"] == "running":
                info["status"] = "timeout"

        return self._get_obs(), reward, done, info

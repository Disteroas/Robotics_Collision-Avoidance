# paper_implementation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create branch `paper_implementation` from `curriculum_learning` implementing Feng et al. 2021 reward, β=0.999, ε reset at Phase 2, and per-episode random spawn via Gazebo teleport.

**Architecture:** Five targeted file edits on top of the existing curriculum infrastructure. `usv_env.py` gains a `_teleport()` method that calls `/gazebo/set_entity_state` after each `/reset_world`. `train.py` resets ε to `max(ε, 0.5)` when Phase 2 activates. Reward is reduced to `+5 / −1000`.

**Tech Stack:** Python 3, ROS 2 Humble, `gazebo_msgs/srv/SetEntityState`, PyTorch, pytest.

---

## File map

| File | Change |
|---|---|
| `src/my_usv/scripts/usv_logic.py` | Remove complex reward constants + simplify `compute_reward` |
| `src/my_usv/test/test_usv_logic.py` | Replace complex-reward tests with simple-reward tests |
| `src/my_usv/scripts/train_core.py` | `BETA_DECAY` 0.995 → 0.999 |
| `src/my_usv/scripts/usv_env.py` | Add `SPAWN_LISTS`, `_teleport()`, `maze_id` param to `reset_environment` |
| `src/my_usv/scripts/train.py` | Add `EPSILON_RESET_P2`, apply reset on Phase 2, pass `maze_id` to `env.reset_environment()` |
| `start_training_curriculum.sh` | Add comment clarifying SPAWN[] is initial-launch-only; per-episode spawn in `usv_env.py` |

---

## Task 1: Create branch

**Files:** git operations only

- [ ] **Step 1: Create and switch to new branch**

```bash
git checkout curriculum_learning
git checkout -b paper_implementation
```

Expected: `Switched to a new branch 'paper_implementation'`

- [ ] **Step 2: Verify branch**

```bash
git branch
```

Expected: `* paper_implementation` in output.

- [ ] **Step 3: Initial commit marking branch start**

```bash
git commit --allow-empty -m "chore: init paper_implementation branch from curriculum_learning"
```

---

## Task 2: Simplify reward in `usv_logic.py` + update tests

**Files:**
- Modify: `src/my_usv/scripts/usv_logic.py`
- Modify: `src/my_usv/test/test_usv_logic.py`

### Step 2a — update tests first (TDD)

- [ ] **Step 1: Replace test_usv_logic.py**

Replace the entire file content with:

```python
import pytest
import numpy as np
from usv_logic import (
    process_lidar, compute_reward,
    LIDAR_MAX_RANGE, LIDAR_BEAMS, COLLISION_DIST,
)


# ─────────────────────────────────────────────────────────────────
# process_lidar  (unchanged logic — tests kept as-is)
# ─────────────────────────────────────────────────────────────────

def test_output_is_exactly_50_bins():
    result = process_lidar([3.0] * 512)
    assert result.shape == (50,)


def test_nan_replaced_with_max_range():
    result = process_lidar([float('nan')] * 512)
    assert np.all(result == LIDAR_MAX_RANGE)


def test_pos_inf_replaced_with_max_range():
    result = process_lidar([float('inf')] * 512)
    assert np.all(result == LIDAR_MAX_RANGE)


def test_values_above_max_clipped():
    result = process_lidar([99.0] * 512)
    assert np.all(result == LIDAR_MAX_RANGE)


def test_all_output_values_in_valid_range():
    raw = [1.0, 2.0, float('nan'), float('inf'), -1.0] * 103
    result = process_lidar(raw[:512])
    assert np.all(result >= 0.0)
    assert np.all(result <= LIDAR_MAX_RANGE)


def test_min_pooling_picks_nearest_obstacle_in_bin():
    raw = [LIDAR_MAX_RANGE] * 512
    raw[5] = 0.5
    result = process_lidar(raw)
    assert result[0] == pytest.approx(0.5)
    assert np.all(result[1:] == LIDAR_MAX_RANGE)


def test_obstacle_in_last_bin_detected():
    raw = [LIDAR_MAX_RANGE] * 512
    raw[-1] = 0.3
    result = process_lidar(raw)
    assert result[-1] == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────
# compute_reward  (Feng et al. 2021 — pure +5 / -1000)
# ─────────────────────────────────────────────────────────────────

def _clear_scan():
    return np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE


def test_collision_returns_minus_1000_and_done():
    scan = np.ones(LIDAR_BEAMS) * (COLLISION_DIST - 0.01)
    reward, done = compute_reward(scan, action_index=5)
    assert reward == -1000.0
    assert done is True


def test_collision_triggered_by_single_ray_below_threshold():
    scan = _clear_scan()
    scan[20] = COLLISION_DIST - 0.01
    reward, done = compute_reward(scan, action_index=5)
    assert reward == -1000.0
    assert done is True


def test_no_collision_returns_exactly_5():
    reward, done = compute_reward(_clear_scan(), action_index=5)
    assert reward == pytest.approx(5.0)
    assert done is False


def test_action_index_does_not_affect_reward():
    scan = _clear_scan()
    rewards = [compute_reward(scan, action_index=i)[0] for i in range(11)]
    assert all(r == pytest.approx(5.0) for r in rewards)


def test_obstacle_far_away_no_collision():
    scan = np.ones(LIDAR_BEAMS) * (COLLISION_DIST + 0.01)
    reward, done = compute_reward(scan, action_index=5)
    assert reward == pytest.approx(5.0)
    assert done is False


def test_reward_at_exact_collision_boundary():
    scan = np.ones(LIDAR_BEAMS) * COLLISION_DIST
    reward, done = compute_reward(scan, action_index=5)
    # min(scan) == COLLISION_DIST is NOT < COLLISION_DIST → no collision
    assert reward == pytest.approx(5.0)
    assert done is False
```

- [ ] **Step 2: Run tests — expect FAIL on reward tests (old imports broken)**

```bash
cd src/my_usv && python -m pytest test/test_usv_logic.py -v 2>&1 | tail -20
```

Expected: `ImportError: cannot import name 'FRONT_DANGER'` or similar — confirms tests drive the change.

### Step 2b — implement simplified reward

- [ ] **Step 3: Replace usv_logic.py**

Replace the entire file content with:

```python
import numpy as np

LIDAR_MAX_RANGE = 5.0
LIDAR_BEAMS     = 50
COLLISION_DIST  = 0.25
LINEAR_VEL      = 0.5


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    chunks = np.array_split(scan, n_bins)
    return np.array([np.min(chunk) for chunk in chunks])


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False
```

- [ ] **Step 4: Run tests — expect all PASS**

```bash
cd src/my_usv && python -m pytest test/test_usv_logic.py -v
```

Expected: all green, 0 failures.

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/usv_logic.py src/my_usv/test/test_usv_logic.py
git commit -m "feat(reward): simplify to +5/-1000 per Feng et al. 2021"
```

---

## Task 3: Update BETA_DECAY in `train_core.py`

**Files:**
- Modify: `src/my_usv/scripts/train_core.py:24`
- Test: `src/my_usv/test/test_agent.py` (no change needed — test imports `BETA_DECAY` and checks it's applied; value change is enough)

- [ ] **Step 1: Run existing agent test to confirm current value**

```bash
cd src/my_usv && python -m pytest test/test_agent.py::test_epsilon_decays_by_beta_each_episode -v
```

Expected: PASS (confirms test structure is sound before change).

- [ ] **Step 2: Change BETA_DECAY**

In `src/my_usv/scripts/train_core.py`, change line 24:

```python
# Before:
BETA_DECAY          = 0.995

# After:
BETA_DECAY          = 0.999
```

- [ ] **Step 3: Run agent tests**

```bash
cd src/my_usv && python -m pytest test/test_agent.py -v
```

Expected: all PASS. `test_epsilon_decays_by_beta_each_episode` now verifies decay with 0.999.

- [ ] **Step 4: Commit**

```bash
git add src/my_usv/scripts/train_core.py
git commit -m "feat(epsilon): BETA_DECAY 0.995 → 0.999 for full 3000-ep decay curve"
```

---

## Task 4: Add spawn randomization to `usv_env.py`

**Files:**
- Modify: `src/my_usv/scripts/usv_env.py`

No unit test possible without a live ROS 2 / Gazebo instance. Validation is done via `./test_spawns.sh` after all tasks are complete.

- [ ] **Step 1: Replace usv_env.py with new version**

Replace the entire file content with:

```python
import math
import random
import rclpy
import rclpy.parameter
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
import numpy as np

from usv_logic import process_lidar, compute_reward, LIDAR_MAX_RANGE, LIDAR_BEAMS, LINEAR_VEL

# Random spawn positions per maze — validate with ./test_spawns.sh before training
SPAWN_LISTS = {
    1: [
        (-3.0, -5.0,  1.57),  # M1-A1: south open, heading N
        ( 0.0, -4.5,  1.57),  # M1-A2: centre-south, heading N
        ( 2.5, -5.0,  1.57),  # M1-A3: right-south, heading N
        (-1.5, -5.0,  0.0 ),  # M1-A4: south open, heading E
        (-2.9, -2.0,  1.57),  # M1-B1: left channel entry, heading N
        (-2.9,  0.5,  0.0 ),  # M1-B2: left channel mid, heading E
        ( 2.5, -2.0,  1.57),  # M1-C1: right outer, heading N
        ( 0.5, -2.5,  1.57),  # M1-D1: centre-bottom, heading N
    ],
    2: [
        (-6.0,  0.0,  0.0 ),  # M2-A1: left entrance, heading E
        (-6.0, -1.5,  0.0 ),  # M2-A2: lower-left, heading E
        (-6.0,  2.0,  0.0 ),  # M2-A3: upper-left, heading E
        (-6.0,  0.0,  1.57),  # M2-A4: left entrance, heading N
        (-3.5,  0.5,  0.0 ),  # M2-B1: centre-left, heading E
        (-3.5, -2.5,  1.57),  # M2-B2: centre-left low, heading N
        (-1.5, -2.5,  0.0 ),  # M2-C1: between Wall_32/Wall_20, heading E
        ( 1.5,  0.0,  3.14),  # M2-D1: right-centre, heading W
    ],
}


class UsvEnv(Node):

    def __init__(self):
        super().__init__(
            'usv_rl_environment',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time', rclpy.Parameter.Type.BOOL, True
                )
            ]
        )

        self.vel_pub        = self.create_publisher(Twist, 'cmd_vel', 10)
        self.scan_sub       = self.create_subscription(LaserScan, 'scan', self._scan_cb, 10)
        self.reset_client   = self.create_client(Empty, '/reset_world')
        self.teleport_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')

        self.current_scan   = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self.accepting_scans = True
        self._lidar_checked = False

        self.get_logger().info("Attendo clock simulato di Gazebo...")
        while self.get_clock().now().nanoseconds == 0:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("Clock simulato attivo.")

    # ──────────────────────────────────────────────────────────────
    # CLOCK SIMULATO
    # ──────────────────────────────────────────────────────────────
    def _wait_sim_seconds(self, sim_sec: float) -> None:
        start_time = self.get_clock().now()
        wait_duration = rclpy.time.Duration(seconds=sim_sec)
        target_time = start_time + wait_duration
        while self.get_clock().now() < target_time:
            rclpy.spin_once(self, timeout_sec=0.001)

    # ──────────────────────────────────────────────────────────────
    # TELEPORT
    # ──────────────────────────────────────────────────────────────
    def _teleport(self, x: float, y: float, yaw: float) -> None:
        while not self.teleport_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /gazebo/set_entity_state...")
        req = SetEntityState.Request()
        req.state.name = 'usv_robot'
        req.state.pose.position.x = float(x)
        req.state.pose.position.y = float(y)
        req.state.pose.position.z = 0.0
        req.state.pose.orientation.x = 0.0
        req.state.pose.orientation.y = 0.0
        req.state.pose.orientation.z = math.sin(yaw / 2.0)
        req.state.pose.orientation.w = math.cos(yaw / 2.0)
        req.state.twist.linear.x  = 0.0
        req.state.twist.angular.z = 0.0
        future = self.teleport_client.call_async(req)
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.01)
        self._wait_sim_seconds(0.3)

    # ──────────────────────────────────────────────────────────────
    # RESET
    # ──────────────────────────────────────────────────────────────
    def reset_environment(self, maze_id: int = 1) -> np.ndarray:
        self.vel_pub.publish(Twist())
        self.accepting_scans = False
        self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE
        self._lidar_checked = False

        while not self.reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("Attendo /reset_world...")

        future = self.reset_client.call_async(Empty.Request())
        while not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
        self._teleport(x, y, yaw)

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.0)

        self._wait_sim_seconds(0.8)
        self.accepting_scans = True

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.get_state()

    # ──────────────────────────────────────────────────────────────
    # SCAN CALLBACK
    # ──────────────────────────────────────────────────────────────
    def _scan_cb(self, msg: LaserScan) -> None:
        if not self.accepting_scans:
            return

        if not self._lidar_checked:
            import math as _math
            min_deg = _math.degrees(msg.angle_min)
            max_deg = _math.degrees(msg.angle_max)
            self.get_logger().info(
                f"LIDAR INFO: Ricevuti {len(msg.ranges)} raggi | "
                f"FOV=[{min_deg:.1f}°, {max_deg:.1f}°]"
            )
            self._lidar_checked = True

        self.current_scan = process_lidar(msg.ranges)

    # ──────────────────────────────────────────────────────────────
    # STEP
    # ──────────────────────────────────────────────────────────────
    def step_action(self, action_index: int):
        cmd = Twist()
        cmd.linear.x  = LINEAR_VEL
        cmd.angular.z = -0.8 + 0.16 * action_index
        self.vel_pub.publish(cmd)

        self._wait_sim_seconds(0.1)
        rclpy.spin_once(self, timeout_sec=0.05)

        reward, done = compute_reward(self.current_scan, action_index)
        return self.get_state(), reward, done

    def get_state(self) -> np.ndarray:
        return (self.current_scan / LIDAR_MAX_RANGE).copy()
```

- [ ] **Step 2: Commit**

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "feat(spawn): per-episode random spawn via SetEntityState teleport"
```

---

## Task 5: Add ε reset + maze_id to `train.py`

**Files:**
- Modify: `src/my_usv/scripts/train.py`

- [ ] **Step 1: Add EPSILON_RESET_P2 constant**

In `src/my_usv/scripts/train.py`, after the existing constants block (after line with `PHASE1_WINDOW`), add:

```python
EPSILON_RESET_P2  = 0.5    # floor ε when Phase 2 activates (Narvekar et al. 2020)
```

The constants block should look like:

```python
MAX_STEPS = 1000
PHASE2_THRESHOLD  = 1500
PHASE1_WINDOW     = 50
EPSILON_RESET_P2  = 0.5    # floor ε when Phase 2 activates (Narvekar et al. 2020)
```

- [ ] **Step 2: Pass maze_id to env.reset_environment()**

Find the line inside the episode loop:

```python
        state  = env.reset_environment()
```

Change it to:

```python
        state  = env.reset_environment(maze_id=args.maze_id)
```

- [ ] **Step 3: Add ε reset on Phase 2 activation**

Find the block inside the episode loop that writes phase 2:

```python
                if not os.path.exists(phase_path) or open(phase_path).read().strip() == '1':
                    _write_phase(phase_path, 2)
                    print(f"  PHASE 2 sbloccata! avg50_maze1={float(np.mean(maze1_window)):.1f} > {PHASE2_THRESHOLD}")
```

Replace with:

```python
                if not os.path.exists(phase_path) or open(phase_path).read().strip() == '1':
                    _write_phase(phase_path, 2)
                    agent.epsilon = max(agent.epsilon, EPSILON_RESET_P2)
                    print(f"  PHASE 2 sbloccata! avg50_maze1={float(np.mean(maze1_window)):.1f} > {PHASE2_THRESHOLD}")
                    print(f"  ε reset → {agent.epsilon:.3f}")
```

- [ ] **Step 4: Verify train.py syntax**

```bash
python -c "import ast; ast.parse(open('src/my_usv/scripts/train.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "feat(epsilon): reset to max(eps, 0.5) on Phase 2 + pass maze_id to env.reset()"
```

---

## Task 6: Update `start_training_curriculum.sh` comment

**Files:**
- Modify: `start_training_curriculum.sh`

The `SPAWN[]` array is still needed as the **initial Gazebo launch position** (where `/reset_world` returns the robot before teleport). No functional change needed — just clarify intent.

- [ ] **Step 1: Update comment on SPAWN array**

Find the lines:

```bash
SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"
```

Replace with:

```bash
# Posizione iniziale lancio Gazebo (punto di ritorno di /reset_world).
# Per-episode spawn randomizzato in usv_env.py via _teleport() — non modificare qui.
SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"
```

- [ ] **Step 2: Update header banner curriculum line**

Find the banner line:

```bash
printf "║  Curriculum         : %-40s║\n" "Phase1=maze1 | Phase2=30/70 (thr:avg50>1500)"
```

Replace with:

```bash
printf "║  Curriculum         : %-40s║\n" "Phase1=M1 | Phase2=30/70 | eps_reset=0.5"
```

- [ ] **Step 3: Commit**

```bash
git add start_training_curriculum.sh
git commit -m "docs(train-script): clarify SPAWN[] is launch-only; per-episode spawn in usv_env.py"
```

---

## Task 7: Run full test suite + spawn validation

- [ ] **Step 1: Run all unit tests**

```bash
cd src/my_usv && python -m pytest test/ -v
```

Expected: all green. If any test imports `FRONT_DANGER` or `SIDE_DANGER` and fails, that import must be removed from that test file.

- [ ] **Step 2: Verify train.py + usv_env.py import cleanly**

```bash
python -c "
import ast
for f in ['src/my_usv/scripts/train.py',
          'src/my_usv/scripts/usv_env.py',
          'src/my_usv/scripts/usv_logic.py',
          'src/my_usv/scripts/train_core.py']:
    ast.parse(open(f).read())
    print(f'OK: {f}')
"
```

Expected: 4 `OK:` lines.

- [ ] **Step 3: Run spawn validator (requires Docker + Gazebo)**

```bash
./test_spawns.sh 1
```

Wait ~7 minutes. Check output. Any `❌ COLLISION` spawn must be removed from `SPAWN_LISTS` in `usv_env.py` and from `SPAWNS_1` in `test_spawns.sh`.

```bash
./test_spawns.sh 2
```

Wait ~7 minutes. Same — remove any `❌ COLLISION` entries from `SPAWN_LISTS[2]` and `SPAWNS_2`.

- [ ] **Step 4: Commit spawn list adjustments if any removals were needed**

```bash
git add src/my_usv/scripts/usv_env.py test_spawns.sh
git commit -m "fix(spawn): remove colliding spawn points found by test_spawns.sh"
```

(Skip this step if all spawns passed.)

- [ ] **Step 5: Final summary commit**

```bash
git log --oneline -8
```

Expected output (approximately):

```
xxxxxxx fix(spawn): remove colliding spawn points ...   (if needed)
xxxxxxx docs(train-script): clarify SPAWN[] is launch-only
xxxxxxx feat(epsilon): reset to max(eps, 0.5) on Phase 2
xxxxxxx feat(spawn): per-episode random spawn via SetEntityState teleport
xxxxxxx feat(epsilon): BETA_DECAY 0.995 → 0.999
xxxxxxx feat(reward): simplify to +5/-1000 per Feng et al. 2021
xxxxxxx chore: init paper_implementation branch from curriculum_learning
```

---

## Self-review checklist

**Spec coverage:**
- ✅ Branch from `curriculum_learning` → Task 1
- ✅ Reward `+5/−1000` → Task 2
- ✅ `BETA_DECAY=0.999` → Task 3
- ✅ `SPAWN_LISTS` + `_teleport()` → Task 4
- ✅ `reset_environment(maze_id)` → Task 4 + Task 5
- ✅ `EPSILON_RESET_P2=0.5` with `max()` → Task 5
- ✅ `start_training_curriculum.sh` comment update → Task 6
- ✅ Spawn validation → Task 7

**Placeholder scan:** No TBD or TODO in plan.

**Type consistency:**
- `reset_environment(maze_id: int = 1)` defined in Task 4, called with `maze_id=args.maze_id` in Task 5 ✓
- `SPAWN_LISTS` dict keys `1` and `2` match `maze_id` values passed from CLI ✓
- `agent.epsilon = max(agent.epsilon, EPSILON_RESET_P2)` — both `float`, no type mismatch ✓
- `_teleport(x, y, yaw)` called with `*random.choice(SPAWN_LISTS[maze_id])` which unpacks a 3-tuple ✓

**Edge case:** `random.choice(SPAWN_LISTS[maze_id])` — `maze_id=3` (test maze) is never passed during training (shell script never selects maze 3 for training blocks). `SPAWN_LISTS` does not define key `3` intentionally — test.py uses fixed spawn, not `reset_environment()`.

# Spawn Diversity & Safety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 8 clustered spawn points on Maze 2 with 16 well-distributed points covering 6 zones, and add post-teleport safety validation to eliminate -1000 crashes at step=1.

**Architecture:** Two changes in `usv_env.py` — a new 16-entry `SPAWN_LISTS[2]` covering zones A–F with 6 yaw orientations, and a retry loop in `reset_environment` that re-teleports if min LIDAR < 0.40m after spawn. Both guarded by tests in a new `test_usv_env.py`. A mandatory `test_spawns.sh` validation gate confirms all 16 coordinates are physically safe in Gazebo before training.

**Tech Stack:** Python, numpy, ROS2 rclpy, pytest, Docker, Gazebo

**Branch:** `feng_direct`

---

## File Structure

| File | Action |
|------|--------|
| `src/my_usv/scripts/usv_env.py` | Modify — add constants, replace SPAWN_LISTS[2], replace reset_environment spawn block |
| `src/my_usv/test/test_usv_env.py` | Create — unit tests for constants, spawn list structure, zone coverage |

---

## Task 1: Spawn list tests + replace SPAWN_LISTS[2]

**Files:**
- Create: `src/my_usv/test/test_usv_env.py`
- Modify: `src/my_usv/scripts/usv_env.py:26-35`

- [ ] **Step 1: Create test file with 4 tests**

Create `src/my_usv/test/test_usv_env.py`:

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

import pytest
from usv_env import SPAWN_LISTS


def test_maze2_spawn_count():
    assert len(SPAWN_LISTS[2]) == 16


def test_all_maze2_spawns_have_three_floats():
    for entry in SPAWN_LISTS[2]:
        x, y, yaw = entry
        assert isinstance(float(x), float)
        assert isinstance(float(y), float)
        assert isinstance(float(yaw), float)


def test_maze2_covers_six_zones():
    spawns = SPAWN_LISTS[2]
    zone_a = [s for s in spawns if s[0] <= -5.5]
    zone_b = [s for s in spawns if -5.5 < s[0] <= -4.0]
    zone_c = [s for s in spawns if -4.0 < s[0] <= -1.0 and s[1] > -3.0 and s[1] < 2.5]
    zone_d = [s for s in spawns if s[0] > -1.0 and s[1] > -3.0 and s[1] < 2.5]
    zone_e = [s for s in spawns if s[1] >= 2.5]
    zone_f = [s for s in spawns if s[1] <= -3.0]
    assert len(zone_a) >= 2, f"Zone A needs >=2 spawns, got {len(zone_a)}"
    assert len(zone_b) >= 2, f"Zone B needs >=2 spawns, got {len(zone_b)}"
    assert len(zone_c) >= 2, f"Zone C needs >=2 spawns, got {len(zone_c)}"
    assert len(zone_d) >= 2, f"Zone D needs >=2 spawns, got {len(zone_d)}"
    assert len(zone_e) >= 2, f"Zone E (upper) needs >=2 spawns, got {len(zone_e)}"
    assert len(zone_f) >= 2, f"Zone F (lower) needs >=2 spawns, got {len(zone_f)}"


def test_maze2_has_diagonal_orientations():
    CARDINAL = {0.0, 1.571, 3.142, 4.712}
    diagonal = [
        s for s in SPAWN_LISTS[2]
        if not any(abs(s[2] - c) < 0.1 for c in CARDINAL)
    ]
    assert len(diagonal) >= 2, f"Need >=2 diagonal yaw spawns, got {len(diagonal)}"
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && source install/setup.bash && \
    python3 -m pytest src/my_usv/test/test_usv_env.py -v"
```

Expected output (current 8-point list):
```
FAILED test_usv_env.py::test_maze2_spawn_count - AssertionError: assert 8 == 16
FAILED test_usv_env.py::test_maze2_covers_six_zones - Zone E (upper) needs >=2 spawns, got 0
FAILED test_usv_env.py::test_maze2_covers_six_zones - Zone F (lower) needs >=2 spawns, got 0
FAILED test_usv_env.py::test_maze2_has_diagonal_orientations - Need >=2 diagonal yaw spawns, got 0
```

- [ ] **Step 3: Replace SPAWN_LISTS[2] in usv_env.py**

In `src/my_usv/scripts/usv_env.py`, replace lines 26–35:

```python
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
```

with:

```python
    2: [
        # Zone A: ingresso sinistro (2) — x <= -5.5
        (-6.0,  0.0,  0.0  ),  # A1: heading E  [validated OK]
        (-6.0,  2.0,  4.712),  # A2: heading S  [validated OK, new yaw]
        # Zone B: centro-sinistra (3) — -5.5 < x <= -4.0
        (-4.5,  0.5,  0.0  ),  # B1: heading E
        (-4.5, -1.5,  1.571),  # B2: heading N  [replaces WARNING at -3.5,-2.5]
        (-5.0,  1.5,  2.356),  # B3: heading NW (135 deg)
        # Zone C: centro (3) — -4.0 < x <= -1.0, y in (-3.0, 2.5)
        (-2.5,  1.0,  0.0  ),  # C1: heading E
        (-1.5, -2.5,  0.0  ),  # C2: heading E  [validated OK]
        (-2.0, -1.0,  0.785),  # C3: heading NE (45 deg)
        # Zone D: centro-destra (3) — x > -1.0, y in (-3.0, 2.5)
        ( 1.5,  0.0,  3.142),  # D1: heading W  [validated OK]
        ( 1.5, -1.5,  1.571),  # D2: heading N
        ( 2.0,  1.0,  4.712),  # D3: heading S
        # Zone E: superiore (2) — y >= 2.5  [zero coverage in old list]
        (-3.0,  3.0,  0.0  ),  # E1: heading E
        ( 0.0,  3.5,  3.142),  # E2: heading W
        # Zone F: inferiore (3) — y <= -3.0  [zero coverage in old list]
        (-4.5, -3.5,  0.0  ),  # F1: heading E
        (-1.5, -4.0,  1.571),  # F2: heading N
        ( 0.5, -3.5,  3.142),  # F3: heading W
    ],
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && source install/setup.bash && \
    python3 -m pytest src/my_usv/test/test_usv_env.py -v"
```

Expected:
```
PASSED test_usv_env.py::test_maze2_spawn_count
PASSED test_usv_env.py::test_all_maze2_spawns_have_three_floats
PASSED test_usv_env.py::test_maze2_covers_six_zones
PASSED test_usv_env.py::test_maze2_has_diagonal_orientations
4 passed in <1s
```

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/usv_env.py src/my_usv/test/test_usv_env.py
git commit -m "feat(spawn): expand maze 2 spawn list to 16 points across 6 zones"
```

---

## Task 2: Safety constants tests + spawn retry loop

**Files:**
- Modify: `src/my_usv/test/test_usv_env.py` — add 2 constant tests
- Modify: `src/my_usv/scripts/usv_env.py:36` — add constants after SPAWN_LISTS
- Modify: `src/my_usv/scripts/usv_env.py:113-125` — replace spawn block with retry loop

- [ ] **Step 1: Add constant tests to test_usv_env.py**

Append to `src/my_usv/test/test_usv_env.py`:

```python
from usv_env import SPAWN_LISTS, SPAWN_SAFETY_DIST, SPAWN_MAX_RETRIES


def test_spawn_safety_dist_is_0_40():
    assert SPAWN_SAFETY_DIST == pytest.approx(0.40)


def test_spawn_max_retries_is_3():
    assert SPAWN_MAX_RETRIES == 3
```

Note: the `from usv_env import ...` line at the top of the file must be updated to include the new names. Replace the existing import line:

```python
from usv_env import SPAWN_LISTS
```

with:

```python
from usv_env import SPAWN_LISTS, SPAWN_SAFETY_DIST, SPAWN_MAX_RETRIES
```

- [ ] **Step 2: Run tests — expect FAIL on new constants**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && source install/setup.bash && \
    python3 -m pytest src/my_usv/test/test_usv_env.py -v"
```

Expected:
```
PASSED test_usv_env.py::test_maze2_spawn_count
PASSED test_usv_env.py::test_all_maze2_spawns_have_three_floats
PASSED test_usv_env.py::test_maze2_covers_six_zones
PASSED test_usv_env.py::test_maze2_has_diagonal_orientations
FAILED test_usv_env.py::test_spawn_safety_dist_is_0_40 - ImportError: cannot import name 'SPAWN_SAFETY_DIST'
FAILED test_usv_env.py::test_spawn_max_retries_is_3 - ImportError: cannot import name 'SPAWN_MAX_RETRIES'
```

- [ ] **Step 3: Add constants to usv_env.py**

In `src/my_usv/scripts/usv_env.py`, after the closing `}` of `SPAWN_LISTS` (after line 36), insert:

```python
SPAWN_SAFETY_DIST = 0.40   # min LIDAR (m) required after teleport
SPAWN_MAX_RETRIES = 3      # max re-teleport attempts if spawn too close to wall
```

Full context around the insertion point:

```python
        ( 0.5, -3.5,  3.142),  # F3: heading W
    ],
}

SPAWN_SAFETY_DIST = 0.40   # min LIDAR (m) required after teleport
SPAWN_MAX_RETRIES = 3      # max re-teleport attempts if spawn too close to wall


class UsvEnv(Node):
```

- [ ] **Step 4: Run tests — expect PASS on constants**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && source install/setup.bash && \
    python3 -m pytest src/my_usv/test/test_usv_env.py -v"
```

Expected:
```
6 passed in <1s
```

- [ ] **Step 5: Replace spawn block in reset_environment**

In `src/my_usv/scripts/usv_env.py`, replace lines 113–125:

```python
        x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
        self._teleport(x, y, yaw)

        for _ in range(20):
            rclpy.spin_once(self, timeout_sec=0.0)

        self._wait_sim_seconds(0.8)
        self.accepting_scans = True

        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.1)

        return self.get_state()
```

with:

```python
        for attempt in range(SPAWN_MAX_RETRIES):
            x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
            self._teleport(x, y, yaw)

            for _ in range(20):
                rclpy.spin_once(self, timeout_sec=0.0)

            self._wait_sim_seconds(0.8)
            self.accepting_scans = True

            for _ in range(5):
                rclpy.spin_once(self, timeout_sec=0.1)

            min_dist = float(self.current_scan.min()) * LIDAR_MAX_RANGE
            if min_dist >= SPAWN_SAFETY_DIST:
                break

            self.get_logger().warn(
                f"Spawn ({x:.1f},{y:.1f}) unsafe: min={min_dist:.2f}m < "
                f"{SPAWN_SAFETY_DIST}m, retry {attempt + 1}/{SPAWN_MAX_RETRIES}"
            )
            self.accepting_scans = False
            self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

        return self.get_state()
```

- [ ] **Step 6: Run full test suite — expect all PASS**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && source install/setup.bash && \
    python3 -m pytest src/my_usv/test/ -v"
```

Expected: all existing tests in `test_usv_logic.py` still pass + 6 new tests pass.

```
13 passed (test_usv_logic.py) + 6 passed (test_usv_env.py) = 19 passed
```

- [ ] **Step 7: Commit**

```bash
git add src/my_usv/scripts/usv_env.py src/my_usv/test/test_usv_env.py
git commit -m "feat(spawn): add post-teleport safety validation with retry loop"
```

---

## Task 3: Spawn validation gate (OBBLIGATORIO prima del training)

**Files:**
- Possibly modify: `src/my_usv/scripts/usv_env.py:26-41` — adjust coordinates if test_spawns finds issues

This task validates all 16 spawn points physically in Gazebo. Some coordinates are new and have not been tested. Any COLLISION or repeated WARNING must be fixed before training.

- [ ] **Step 1: Build to propagate updated world + scripts to install/**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
```

Expected: `Summary: 1 package finished`

- [ ] **Step 2: Run spawn validator on Maze 2**

```bash
./test_spawns.sh 2
```

Expected (all 16 pass): output shows 16 rows, each ✅ OK or ⚠️ WARNING.
Runtime: ~8 minutes (16 spawns × ~30s each).

```
╔══════════════════════════════════════════════════════════════╗
║  ✅  OK: 14   ⚠️   WARN: 2   ❌  FAIL: 0   ⏱️  TO: 0   ║
╚══════════════════════════════════════════════════════════════╝
```

**Acceptance criteria:**
- Zero ❌ COLLISION — if any, move that coordinate 0.5m in the direction with most LIDAR clearance
- Zero ⏱️ TIMEOUT — if any, re-run; if persistent, increase `GAZEBO_WAIT` in `test_spawns.sh`
- ⚠️ WARNING is acceptable (0.25–0.40m) but ideally move to ≥0.40m — the safety retry loop handles these at runtime

- [ ] **Step 3: Fix any COLLISION spawn**

If a spawn shows ❌ COLLISION, identify which zone it belongs to from the label and move it.

Example fix for F1 (-4.5, -3.5) if it collides:

In `src/my_usv/scripts/usv_env.py`, change:

```python
        (-4.5, -3.5,  0.0  ),  # F1: heading E
```

to (move +0.5m in x, away from left boundary):

```python
        (-4.0, -3.5,  0.0  ),  # F1: heading E  [moved from -4.5 due to collision]
```

Then re-run:

```bash
./test_spawns.sh 2
```

Repeat until zero COLLISION.

- [ ] **Step 4: Commit if any coordinates changed**

Only if any coordinates were adjusted in Step 3:

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "fix(spawn): adjust maze 2 spawn coordinates after test_spawns validation"
```

If zero changes needed: no commit required.

- [ ] **Step 5: Run tests to confirm spawn list structure still valid**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && source install/setup.bash && \
    python3 -m pytest src/my_usv/test/ -v"
```

Expected: 19 passed. If a zone test fails because a moved coordinate left a zone boundary, adjust the coordinate to stay within the intended zone.

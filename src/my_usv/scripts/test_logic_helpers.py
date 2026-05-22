import numpy as np
from usv_logic import (
    sector_distances, crash_sector, round_robin_spawn, compute_reward,
)


def test_sector_distances_picks_min_per_sector():
    scan = np.full(50, 5.0, dtype=np.float32)
    scan[5] = 0.4    # right  [0:20]
    scan[25] = 0.3   # front  [20:30]
    scan[40] = 0.5   # left   [30:50]
    d = sector_distances(scan)
    assert abs(d['right'] - 0.4) < 1e-6
    assert abs(d['front'] - 0.3) < 1e-6
    assert abs(d['left'] - 0.5) < 1e-6
    assert abs(d['min_lidar'] - 0.3) < 1e-6


def test_crash_sector_returns_closest():
    assert crash_sector(front=0.2, left=0.5, right=0.4) == 'front'
    assert crash_sector(front=0.5, left=0.2, right=0.4) == 'left'
    assert crash_sector(front=0.5, left=0.4, right=0.2) == 'right'


def test_round_robin_cycles():
    spawns = list(range(6))
    seq = [round_robin_spawn(spawns, i) for i in range(7)]
    assert seq == [0, 1, 2, 3, 4, 5, 0]


def test_compute_reward_open_space_unchanged():
    scan = np.full(50, 5.0, dtype=np.float32)
    reward, done = compute_reward(scan, 5)
    assert done is False
    assert abs(reward - 7.0) < 1e-6


def test_compute_reward_collision_unchanged():
    scan = np.full(50, 5.0, dtype=np.float32)
    scan[25] = 0.1   # < COLLISION_DIST nel settore frontale
    reward, done = compute_reward(scan, 5)
    assert done is True
    assert reward == -1000.0

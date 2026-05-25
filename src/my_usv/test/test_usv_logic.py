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


def test_uniform_selection_picks_selected_indices():
    idx = np.linspace(0, 511, 50).round().astype(int)
    raw = [LIDAR_MAX_RANGE] * 512
    raw[idx[3]] = 0.5
    result = process_lidar(raw)
    assert result[3] == pytest.approx(0.5)


def test_uniform_selection_ignores_non_selected_rays():
    idx = np.linspace(0, 511, 50).round().astype(int)
    raw = [LIDAR_MAX_RANGE] * 512
    raw[idx[0] + 1] = 0.5   # ray 1: tra idx[0]=0 e idx[1]≈10, NON selezionato
    result = process_lidar(raw)
    assert result[0] == pytest.approx(LIDAR_MAX_RANGE)


def test_obstacle_in_last_bin_detected():
    raw = [LIDAR_MAX_RANGE] * 512
    raw[-1] = 0.3   # idx[-1]=511 è sempre selezionato
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

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


# ─────────────────────────────────────────────────────────────────
# Round 2 (R-alpha) — narrow front sector + reduced weights
# ─────────────────────────────────────────────────────────────────

def test_front_sector_narrow_indices_20_30():
    """Bin 15 (era front [15:35], ora right [0:20]) close obstacle:
       front_dist deve NON triggerare. Verifica che front_dist = min(scan[20:30])."""
    scan = _clear_scan()
    scan[15] = 0.5  # close obstacle bin 15 — in nuovo right sector
    reward, done = compute_reward(scan, action_index=5)
    # front_dist = min(scan[20:30]) = LIDAR_MAX_RANGE → no front penalty
    # right_dist = min(scan[0:20]) = 0.5 > SIDE_DANGER(0.45) → no side penalty
    # mean(scan) = (0.5 + 49*5.0)/50 ≈ 4.91 → bonus ≈ 1.96
    # reward ≈ 5 + 1.96 = 6.96 (no penalty triggered)
    assert done is False
    assert reward > 6.0, f"reward {reward} troppo basso, indica penalty triggered"


def test_front_penalty_max_weight_is_10():
    """Front bin a 0.26m (just above COLLISION_DIST=0.25) deve produrre
       penalty ≤ 10. Conferma weight ridotto 20→10."""
    scan = _clear_scan()
    # bin 20-29 = front (5.4°/bin × 10 bin = 54° ±27° dall'asse)
    scan[20:30] = 0.26
    reward, done = compute_reward(scan, action_index=5)
    # severity = (1.5 - 0.26) / (1.5 - 0.25) = 0.992
    # penalty front = 10 * 0.992^2 ≈ 9.84
    # mean(scan) = (10*0.26 + 40*5.0)/50 = 4.052
    # bonus = 2.0 * 4.052/5.0 = 1.62
    # reward = 5 + 1.62 - 0 - 9.84 ≈ -3.22
    assert done is False
    assert -4.5 < reward < -1.5, f"reward {reward} fuori range atteso [-4.5, -1.5]"


def test_side_penalty_max_weight_is_3():
    """Side bin (bin 0 = right) a 0.26m deve produrre penalty ≤ 3.
       Conferma weight ridotto 5→3."""
    scan = _clear_scan()
    scan[0] = 0.26  # right side close
    reward, done = compute_reward(scan, action_index=5)
    # right_dist = 0.26 < SIDE_DANGER(0.45) → triggers
    # severity = (0.45 - 0.26) / (0.45 - 0.25) = 0.95
    # penalty side = 3 * 0.95^2 ≈ 2.71
    # mean(scan) = (0.26 + 49*5.0)/50 ≈ 4.905
    # bonus = 2.0 * 4.905/5.0 = 1.962
    # reward = 5 + 1.962 - 2.71 ≈ 4.25
    assert done is False
    assert 3.5 < reward < 5.0, f"reward {reward} fuori range atteso [3.5, 5.0]"

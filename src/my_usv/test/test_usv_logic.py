import pytest
import numpy as np
from usv_logic import (
    process_lidar, compute_reward,
    LIDAR_MAX_RANGE, LIDAR_BEAMS, COLLISION_DIST,
    FRONT_DANGER, SIDE_DANGER,
)



# ─────────────────────────────────────────────────────────────────
# process_lidar
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
    raw = [1.0, 2.0, float('nan'), float('inf'), -1.0] * 103  # >512, trim
    result = process_lidar(raw[:512])
    assert np.all(result >= 0.0)
    assert np.all(result <= LIDAR_MAX_RANGE)


def test_min_pooling_picks_nearest_obstacle_in_bin():
    # Bin 0 copre i raggi 0–10 (primo degli 11-ray bin).
    # Mettiamo un ostacolo solo al raggio 5: deve comparire in bin[0], tutti gli altri a 5.0.
    raw = [LIDAR_MAX_RANGE] * 512
    raw[5] = 0.5
    result = process_lidar(raw)
    assert result[0] == pytest.approx(0.5)
    assert np.all(result[1:] == LIDAR_MAX_RANGE)


def test_obstacle_in_last_bin_detected():
    raw = [LIDAR_MAX_RANGE] * 512
    raw[-1] = 0.3  # ultimo raggio → ultimo bin
    result = process_lidar(raw)
    assert result[-1] == pytest.approx(0.3)


# ─────────────────────────────────────────────────────────────────
# compute_reward
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
    scan[20] = COLLISION_DIST - 0.01  # un solo raggio nel settore frontale
    reward, done = compute_reward(scan, action_index=5)
    assert reward == -1000.0
    assert done is True


def test_clear_path_straight_returns_base_reward():
    # action 5, all clear: +5.0 base + 2.0 space bonus (mean=5.0/5.0*2.0) = 7.0
    reward, done = compute_reward(_clear_scan(), action_index=5)
    assert reward == pytest.approx(7.0)
    assert done is False


def test_hard_left_has_steering_penalty():
    # action 0: |0-5| * 0.02 = 0.1 penalty. Open space: space_bonus=2.0
    reward, _ = compute_reward(_clear_scan(), action_index=0)
    assert reward == pytest.approx(6.9)


def test_hard_right_has_same_penalty_as_hard_left():
    r_left, _  = compute_reward(_clear_scan(), action_index=0)
    r_right, _ = compute_reward(_clear_scan(), action_index=10)
    assert r_left == pytest.approx(r_right)


def test_front_danger_reduces_reward():
    scan = _clear_scan()
    scan[15:35] = 2.0  # front < FRONT_DANGER(3.0m)
    reward_danger, done = compute_reward(scan, action_index=5)
    reward_clear, _     = compute_reward(_clear_scan(), action_index=5)
    assert reward_danger < reward_clear
    assert not done


def test_front_danger_severity_is_quadratic():
    # FRONT_DANGER=3.0, midpoint=(3.0+0.25)/2=1.625 → severity=0.5
    # penalty = 20*(0.5**2) = 5.0
    midpoint = (FRONT_DANGER + COLLISION_DIST) / 2   # 1.625 m
    scan = _clear_scan()
    scan[15:35] = midpoint
    # compute expected: mean(scan) = (30*5.0 + 20*1.625) / 50 = 3.65
    expected_space_bonus = 2.0 * ((30 * LIDAR_MAX_RANGE + 20 * midpoint) / LIDAR_BEAMS) / LIDAR_MAX_RANGE
    expected = 5.0 + expected_space_bonus - 5.0  # base + bonus - front_penalty
    reward, _ = compute_reward(scan, action_index=5)
    assert reward == pytest.approx(expected, abs=1e-4)


def test_right_side_danger_reduces_reward():
    scan = _clear_scan()
    scan[0:15] = 0.35  # destra < SIDE_DANGER(0.45m)
    reward_danger, done = compute_reward(scan, action_index=5)
    reward_clear, _     = compute_reward(_clear_scan(), action_index=5)
    assert reward_danger < reward_clear
    assert not done


def test_left_side_danger_reduces_reward():
    scan = _clear_scan()
    scan[35:50] = 0.35  # sinistra < SIDE_DANGER(0.45m)
    reward_danger, done = compute_reward(scan, action_index=5)
    reward_clear, _     = compute_reward(_clear_scan(), action_index=5)
    assert reward_danger < reward_clear
    assert not done


def test_side_danger_severity_is_quadratic():
    # A metà percorso: severity = 0.5 → penalty = 5*(0.5^2) = 1.25
    midpoint = (SIDE_DANGER + COLLISION_DIST) / 2  # 0.35 m
    scan = _clear_scan()
    scan[0:15] = midpoint
    # space_bonus = 2.0 * mean(scan) / LIDAR_MAX_RANGE
    expected_space_bonus = 2.0 * ((15 * midpoint + 35 * LIDAR_MAX_RANGE) / LIDAR_BEAMS) / LIDAR_MAX_RANGE
    reward, _ = compute_reward(scan, action_index=5)
    assert reward == pytest.approx(5.0 + expected_space_bonus - 1.25, abs=1e-4)


def test_both_sides_danger_penalties_sum():
    midpoint = (SIDE_DANGER + COLLISION_DIST) / 2
    scan = _clear_scan()
    scan[0:15]  = midpoint  # destra → -1.25
    scan[35:50] = midpoint  # sinistra → -1.25
    # space_bonus = 2.0 * mean(scan) / LIDAR_MAX_RANGE
    expected_space_bonus = 2.0 * ((30 * midpoint + 20 * LIDAR_MAX_RANGE) / LIDAR_BEAMS) / LIDAR_MAX_RANGE
    reward, _ = compute_reward(scan, action_index=5)
    assert reward == pytest.approx(5.0 + expected_space_bonus - 1.25 - 1.25, abs=1e-4)


def test_space_bonus_increases_with_open_space():
    scan_clear = _clear_scan()                          # mean=5.0 → bonus=2.0
    scan_tight = np.ones(LIDAR_BEAMS) * 4.0            # mean=4.0 → bonus=1.6 (no danger zone)
    r_clear, _ = compute_reward(scan_clear, action_index=5)
    r_tight, _ = compute_reward(scan_tight, action_index=5)
    assert r_clear > r_tight


def test_steering_penalty_reduced_to_0_02():
    # Hard turn (action 0): penalty = |0-5| * 0.02 = 0.1
    r_straight, _ = compute_reward(_clear_scan(), action_index=5)
    r_turn, _     = compute_reward(_clear_scan(), action_index=0)
    assert r_straight - r_turn == pytest.approx(0.1, abs=1e-4)

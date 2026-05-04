import numpy as np

LIDAR_MAX_RANGE    = 5.0
LIDAR_BEAMS        = 50
COLLISION_DIST     = 0.25
FRONT_DANGER       = 3.0    # esteso da 1.5: robot vede muro 15 step prima
SIDE_DANGER        = 0.45
LINEAR_VEL         = 0.5
SPACE_BONUS_WEIGHT = 2.0    # bonus max per spazio aperto


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    chunks = np.array_split(scan, n_bins)
    return np.array([np.min(chunk) for chunk in chunks])


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    # FOV 270° / 50 bin = 5.4°/bin → destra [-135°,-54°], fronte [-54°,+54°], sinistra [+54°,+135°]
    right_dist = float(np.min(scan[0:15]))
    front_dist = float(np.min(scan[15:35]))
    left_dist  = float(np.min(scan[35:50]))

    if min(right_dist, front_dist, left_dist) < COLLISION_DIST:
        return -1000.0, True

    # Open-space bonus: incentiva navigazione lontano dai muri
    space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE

    # Steering: penalità ridotta (anti-oscillazione, non vincolo comportamentale)
    steering_penalty = abs(action_index - 5) * 0.02

    danger_penalty = 0.0

    # Front danger quadratico su zona estesa: segnale più forte a distanza media
    if front_dist < FRONT_DANGER:
        severity = (FRONT_DANGER - front_dist) / (FRONT_DANGER - COLLISION_DIST)
        danger_penalty += 20.0 * (severity ** 2)

    if right_dist < SIDE_DANGER:
        severity = (SIDE_DANGER - right_dist) / (SIDE_DANGER - COLLISION_DIST)
        danger_penalty += 5.0 * (severity ** 2)

    if left_dist < SIDE_DANGER:
        severity = (SIDE_DANGER - left_dist) / (SIDE_DANGER - COLLISION_DIST)
        danger_penalty += 5.0 * (severity ** 2)

    return 5.0 + space_bonus - steering_penalty - danger_penalty, False

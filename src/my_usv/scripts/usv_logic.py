import numpy as np

LIDAR_MAX_RANGE    = 5.0
LIDAR_BEAMS        = 50
COLLISION_DIST     = 0.25
LINEAR_VEL         = 0.5

# Reward shaping parameters (R-alpha Round 2)
# FOV 270° / 50 bin = 5.4°/bin → right [0:20], front [20:30], left [30:50]
FRONT_DANGER       = 1.5    # m — 30 step preavviso (v=0.5m/s, dt=0.1s/step)
SIDE_DANGER        = 0.45   # m — buffer 0.20m sopra COLLISION_DIST
SPACE_BONUS_WEIGHT = 2.0    # max bonus in spazio completamente aperto

# Confini settore (FOV 270° / 50 bin). Condivisi tra reward e logging (DRY).
RIGHT_SLICE = slice(0, 20)    # 108° destra
FRONT_SLICE = slice(20, 30)   # 54° centro
LEFT_SLICE  = slice(30, 50)   # 108° sinistra


def sector_distances(scan: np.ndarray) -> dict:
    """Distanza minima per settore + minimo globale. Scan già processato (50 bin)."""
    return {
        'right':     float(np.min(scan[RIGHT_SLICE])),
        'front':     float(np.min(scan[FRONT_SLICE])),
        'left':      float(np.min(scan[LEFT_SLICE])),
        'min_lidar': float(np.min(scan)),
    }


def crash_sector(front: float, left: float, right: float) -> str:
    """Settore col valore minimo (responsabile del crash)."""
    return min(
        (('front', front), ('left', left), ('right', right)),
        key=lambda kv: kv[1],
    )[0]


def round_robin_spawn(spawn_list, counter: int):
    """Seleziona lo spawn in modo deterministico ciclico."""
    return spawn_list[counter % len(spawn_list)]


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    chunks = np.array_split(scan, n_bins)
    return np.array([np.min(chunk) for chunk in chunks])


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    right_dist = float(np.min(scan[RIGHT_SLICE]))   # 108° destra
    front_dist = float(np.min(scan[FRONT_SLICE]))   # 54° centro
    left_dist  = float(np.min(scan[LEFT_SLICE]))    # 108° sinistra

    if min(right_dist, front_dist, left_dist) < COLLISION_DIST:
        return -1000.0, True

    # Open-space bonus: incentiva spazio aperto, penalizza wall-following loop
    space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE

    # Soft steering penalty: anti-oscillazione (max 0.10, irrilevante vs pericolo)
    steering_penalty = abs(action_index - 5) * 0.02

    danger_penalty = 0.0

    # Front danger quadratic: segnale forte vicino al muro, zero a FRONT_DANGER
    if front_dist < FRONT_DANGER:
        severity = (FRONT_DANGER - front_dist) / (FRONT_DANGER - COLLISION_DIST)
        danger_penalty += 10.0 * (severity ** 2)  # R-alpha Round 2: era 20.0

    # Side danger quadratic: penalizza avvicinamento laterale (simmetrico)
    if right_dist < SIDE_DANGER:
        severity = (SIDE_DANGER - right_dist) / (SIDE_DANGER - COLLISION_DIST)
        danger_penalty += 3.0 * (severity ** 2)  # R-alpha Round 2: era 5.0

    if left_dist < SIDE_DANGER:
        severity = (SIDE_DANGER - left_dist) / (SIDE_DANGER - COLLISION_DIST)
        danger_penalty += 3.0 * (severity ** 2)  # R-alpha Round 2: era 5.0

    return 5.0 + space_bonus - steering_penalty - danger_penalty, False

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
    # Feng 2021 §5.1: 50 misure selezionate UNIFORMEMENTE dai 512 ray, clip [0, max_range].
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    idx = np.linspace(0, len(scan) - 1, n_bins).round().astype(int)
    return scan[idx]


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    # Feng 2021 Eq.4: reward puro. +5 per step senza collisione, -1000 alla collisione.
    # action_index ignorato (nessuna steering penalty). Le slice settore e gli helper
    # sector_distances/crash_sector restano per il logging/eval, non per la reward.
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False

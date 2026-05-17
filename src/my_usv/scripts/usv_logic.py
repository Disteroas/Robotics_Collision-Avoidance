import numpy as np

LIDAR_MAX_RANGE = 5.0
LIDAR_BEAMS     = 50
COLLISION_DIST  = 0.25
LINEAR_VEL      = 0.5


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    
    # Campionamento uniforme di 50 raggi dai 512 originali (Replica esatta del paper)
    indices = np.linspace(0, len(scan) - 1, n_bins, dtype=int)
    return scan[indices]


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False

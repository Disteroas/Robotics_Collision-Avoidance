import numpy as np

LIDAR_MAX_RANGE = 5.0
LIDAR_BEAMS     = 50
COLLISION_DIST  = 0.25
LINEAR_VEL      = 0.5


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    
    # Min-Pooling: divide i raggi in n_bins e prende il minimo per ogni bin.
    # Questo garantisce di NON bucare mai ostacoli sottili tra un raggio e l'altro.
    bin_size = len(scan) // n_bins
    processed_scan = np.zeros(n_bins, dtype=np.float32)
    
    for i in range(n_bins):
        start_idx = i * bin_size
        # Se è l'ultimo bin, prende tutto il resto dell'array
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(scan)
        processed_scan[i] = np.min(scan[start_idx:end_idx])
        
    return processed_scan


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False

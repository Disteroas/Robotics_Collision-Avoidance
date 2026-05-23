import numpy as np

LIDAR_MAX_RANGE = 5.0
LIDAR_BEAMS     = 50
COLLISION_DIST  = 0.25
LINEAR_VEL      = 0.5


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    """
    Preprocessing LIDAR con min-pooling per settore.

    Il paper Feng 2021 descrive "50 range measurements selected evenly",
    che suggerisce subsampling uniforme. Tuttavia, per collision avoidance
    il min-pool è la scelta più conservativa e sicura: riporta sempre
    l'ostacolo più vicino nel settore, non perdendo mai un ostacolo
    (a differenza del subsampling uniforme, che potrebbe campionare
    casualmente il raggio "libero" invece di quello che ha colpito il muro).

    L'effetto di "masking" delle aperture citato in letteratura è reale
    in linea di principio, ma trascurabile per questa geometria:
    ogni bin copre ~5.4° (512 raggi / 50 bin su 270°). Un corridoio
    da 1.5 m visto a 2 m copre ~41° → 7–8 bin lo vedono chiaramente.
    Solo i 2–3 bin di bordo sono parzialmente affetti.

    Cambiare in subsampling uniforme introdurrebbe il rischio opposto
    (perdere un ostacolo se il raggio campionato cade sul lato libero
    di un bin di confine) senza benefici dimostrabili in questo task.
    """
    if raw_ranges is None or len(raw_ranges) == 0:
        return np.ones(n_bins, dtype=np.float32) * max_range

    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)

    bin_size = len(scan) // n_bins
    if bin_size == 0:
        # Caso di emergenza: meno raggi LIDAR di quanto richiesto
        return np.ones(n_bins, dtype=np.float32) * max_range

    processed_scan = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx   = (i + 1) * bin_size if i < n_bins - 1 else len(scan)
        processed_scan[i] = np.min(scan[start_idx:end_idx])

    return np.nan_to_num(processed_scan, nan=max_range).astype(np.float32)


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    """
    Reward identica al paper Feng 2021, eq. (4):
      +5    senza collisione
      -1000 con collisione (termina episodio)
    """
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True

    return 5.0, False

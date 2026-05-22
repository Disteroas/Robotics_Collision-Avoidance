# Base per simulazioni rigorose — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Costruire l'infrastruttura di misura (seed control + eval deterministico + logging per-step + aggregazione multi-seed) che rende le run riproducibili, valutabili e diagnosticabili, senza toccare il comportamento dell'agente.

**Architecture:** Tre pilastri. (1) `seeding.py` centralizza il seed, invocato da `train.py`/`test.py` via `--seed`. (2) `usv_env` usa round-robin deterministico in test e popola `last_info` con le distanze per settore; `usv_logic.py` ospita le funzioni pure condivise (settori, crash-sector, round-robin). (3) `test.py` scrive CSV per-step + crash + summary; `aggregate_seeds.py` riduce N seed a mean±std/IQM/CI. Gli artefatti vivono in `runs/<config>/seed_<S>/`.

**Tech Stack:** Python 3 (numpy, torch, stdlib csv/json), ROS2 Humble + Gazebo (solo per smoke test), Bash, pytest. Tutto gira dentro il container `usv_rl_project`.

---

## Premessa di esecuzione

I test pure-Python girano **dentro il container** (host Windows può non avere numpy/torch):

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws && source install/setup.bash && pytest <path> -v"
```

I test sono "pure" (no ROS): non serve Gazebo. Gli smoke test ROS sono indicati esplicitamente dove servono.

**Non-goal (NON toccare):** reward in `compute_reward`, rete `ddqn_model.py`, iperparametri, action-space. L'unica modifica a `usv_logic.compute_reward` è sostituire slice letterali con costanti **identiche** (no-op, con test di regressione).

---

## File Structure

| File | Responsabilità | Azione |
|---|---|---|
| `src/my_usv/scripts/seeding.py` | Seed globale | Crea |
| `src/my_usv/scripts/usv_logic.py` | Funzioni pure: settori, crash-sector, round-robin | Modifica |
| `src/my_usv/scripts/aggregate_seeds.py` | Aggregazione multi-seed (mean/std/IQM/CI) | Crea |
| `src/my_usv/scripts/usv_env.py` | Round-robin in test + `last_info` | Modifica |
| `src/my_usv/scripts/train_core.py` | Seed nel checkpoint | Modifica |
| `src/my_usv/scripts/train.py` | `--seed` + `crash_sector` in CSV | Modifica |
| `src/my_usv/scripts/test.py` | Logging per-step/crash/summary, nuovi CLI | Riscrive |
| `src/my_usv/scripts/test_seeding.py` | Test seeding | Crea |
| `src/my_usv/scripts/test_logic_helpers.py` | Test funzioni pure | Crea |
| `src/my_usv/scripts/test_aggregate_seeds.py` | Test aggregazione | Crea |
| `start_train_multimaze.sh` | `--seed`/`--config`, layout `runs/`, backup-guard | Modifica |
| `start_test.sh` | `--seed`/`--config`/`--reps`, layout `runs/`, `run_meta.json` | Modifica |
| `.gitignore` | Ignora `runs/` | Modifica |

---

## Task 1: Modulo seeding

**Files:**
- Create: `src/my_usv/scripts/seeding.py`
- Test: `src/my_usv/scripts/test_seeding.py`

- [ ] **Step 1: Scrivi il test che fallisce**

`src/my_usv/scripts/test_seeding.py`:
```python
import random
import numpy as np
import torch
from seeding import set_global_seed


def test_same_seed_reproducible():
    set_global_seed(42)
    a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_global_seed(42)
    b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert a == b


def test_different_seed_differs():
    set_global_seed(1)
    a = random.random()
    set_global_seed(2)
    b = random.random()
    assert a != b
```

- [ ] **Step 2: Esegui il test, deve fallire**

Run: `pytest src/my_usv/scripts/test_seeding.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'seeding'`

- [ ] **Step 3: Implementa `seeding.py`**

`src/my_usv/scripts/seeding.py`:
```python
"""Controllo centralizzato del seed. Nessuna dipendenza ROS.

NOTA: Gazebo (fisica + timing ROS) resta non deterministico. Il seed NON dà
riproducibilità bit-a-bit, ma rende la varianza attribuibile e misurabile
(Henderson 2018). torch.use_deterministic_algorithms NON è attivato: su CPU
dà overhead e la non-determinismo dominante qui è Gazebo, non i kernel torch.
"""
import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 4: Esegui il test, deve passare**

Run: `pytest src/my_usv/scripts/test_seeding.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/seeding.py src/my_usv/scripts/test_seeding.py
git commit -m "feat(seed): modulo set_global_seed riproducibile"
```

---

## Task 2: Funzioni pure condivise in usv_logic

**Files:**
- Modify: `src/my_usv/scripts/usv_logic.py`
- Test: `src/my_usv/scripts/test_logic_helpers.py`

- [ ] **Step 1: Scrivi i test che falliscono**

`src/my_usv/scripts/test_logic_helpers.py`:
```python
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
    # Regressione: reward in spazio aperto deve restare 7.0 (5 + 2*1.0).
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
```

- [ ] **Step 2: Esegui i test, devono fallire**

Run: `pytest src/my_usv/scripts/test_logic_helpers.py -v`
Expected: FAIL con `ImportError: cannot import name 'sector_distances'`

- [ ] **Step 3: Aggiungi costanti e funzioni a `usv_logic.py`**

Inserire dopo la riga `SPACE_BONUS_WEIGHT = 2.0    # max bonus in spazio completamente aperto` (riga 12):
```python

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
```

- [ ] **Step 4: Refactor no-op di `compute_reward` (usa le costanti)**

In `compute_reward`, sostituire SOLO gli indici letterali con le costanti (valori identici):
```python
    right_dist = float(np.min(scan[RIGHT_SLICE]))   # 108° destra
    front_dist = float(np.min(scan[FRONT_SLICE]))   # 54° centro
    left_dist  = float(np.min(scan[LEFT_SLICE]))    # 108° sinistra
```
(erano `scan[0:20]`, `scan[20:30]`, `scan[30:50]` — semanticamente identici.)

- [ ] **Step 5: Esegui i test, devono passare**

Run: `pytest src/my_usv/scripts/test_logic_helpers.py -v`
Expected: PASS (5 passed). I due test di regressione confermano che il reward è invariato.

- [ ] **Step 6: Commit**

```bash
git add src/my_usv/scripts/usv_logic.py src/my_usv/scripts/test_logic_helpers.py
git commit -m "feat(logic): helper settori/crash/round-robin + refactor no-op slice"
```

---

## Task 3: Tool di aggregazione multi-seed

**Files:**
- Create: `src/my_usv/scripts/aggregate_seeds.py`
- Test: `src/my_usv/scripts/test_aggregate_seeds.py`

Schema atteso di `eval_summary.csv` (prodotto da Task 7): colonne `config,seed,maze,episodes,n_success,success_rate,avg_reward,avg_steps`.

- [ ] **Step 1: Scrivi i test che falliscono**

`src/my_usv/scripts/test_aggregate_seeds.py`:
```python
import csv
import numpy as np
from aggregate_seeds import iqm, bootstrap_ci, aggregate, read_summaries


def test_iqm_small_sample_is_mean():
    assert abs(iqm([0.4, 0.5, 0.6]) - 0.5) < 1e-9


def test_iqm_trims_quartiles():
    # n=8: trim i 2 estremi per lato → media di [3,4,5,6] = 4.5
    assert abs(iqm([1, 2, 3, 4, 5, 6, 7, 8]) - 4.5) < 1e-9


def test_bootstrap_ci_constant_values():
    lo, hi = bootstrap_ci([0.5, 0.5, 0.5], n_resamples=200, seed=0)
    assert abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9


def test_aggregate_groups_by_maze(tmp_path):
    def write(p, rows):
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'config', 'seed', 'maze', 'episodes',
                'n_success', 'success_rate', 'avg_reward', 'avg_steps'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    base = {'config': 'x', 'episodes': 10, 'n_success': 4,
            'avg_reward': 100, 'avg_steps': 200}
    p0 = tmp_path / 's0.csv'
    p1 = tmp_path / 's1.csv'
    write(p0, [{**base, 'seed': 0, 'maze': 1, 'success_rate': 0.4}])
    write(p1, [{**base, 'seed': 1, 'maze': 1, 'success_rate': 0.6}])

    rows = read_summaries([str(p0), str(p1)])
    out = aggregate(rows)
    assert len(out) == 1
    r = out[0]
    assert r['maze'] == 1 and r['n_seed'] == 2
    assert abs(r['mean'] - 0.5) < 1e-9
```

- [ ] **Step 2: Esegui i test, devono fallire**

Run: `pytest src/my_usv/scripts/test_aggregate_seeds.py -v`
Expected: FAIL con `ModuleNotFoundError: No module named 'aggregate_seeds'`

- [ ] **Step 3: Implementa `aggregate_seeds.py`**

`src/my_usv/scripts/aggregate_seeds.py`:
```python
"""Aggrega gli eval_summary.csv di N seed in statistiche robuste.

Reporting (Henderson 2018, Agarwal 2021): mai il max. Sempre mean±std,
IQM (Inter-Quartile Mean) e intervallo di confidenza 95% via bootstrap.
Solo numpy + stdlib (niente pandas: non garantito nel container).
"""
import argparse
import csv
import glob
import os

import numpy as np


def iqm(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float('nan')
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    trimmed = v[lo:hi] if hi > lo else v
    return float(np.mean(trimmed))


def bootstrap_ci(values, n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = np.array([
        np.mean(rng.choice(v, size=len(v), replace=True))
        for _ in range(n_resamples)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def read_summaries(paths):
    rows = []
    for p in paths:
        with open(p, newline='') as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def aggregate(rows):
    by_maze = {}
    for r in rows:
        by_maze.setdefault(int(r['maze']), []).append(float(r['success_rate']))
    out = []
    for maze in sorted(by_maze):
        sr = by_maze[maze]
        lo, hi = bootstrap_ci(sr)
        out.append({
            'maze':    maze,
            'n_seed':  len(sr),
            'mean':    float(np.mean(sr)),
            'std':     float(np.std(sr, ddof=1)) if len(sr) > 1 else 0.0,
            'iqm':     iqm(sr),
            'ci_low':  lo,
            'ci_high': hi,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='default')
    p.add_argument('--runs-dir', type=str, default='runs')
    p.add_argument('--output', type=str, required=True)
    args = p.parse_args()

    pattern = os.path.join(args.runs_dir, args.config, 'seed_*', 'eval_summary.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[ERRORE] Nessun eval_summary.csv in {pattern}")
        raise SystemExit(1)

    rows = read_summaries(paths)
    out = aggregate(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'maze', 'n_seed', 'mean', 'std', 'iqm', 'ci_low', 'ci_high'])
        w.writeheader()
        for r in out:
            w.writerow(r)

    print(f"  ✅ Aggregato {len(paths)} seed → {args.output}")
    for r in out:
        print(f"  M{r['maze']}: {r['mean']*100:.1f}% ± {r['std']*100:.1f}  "
              f"(IQM {r['iqm']*100:.1f}%; 95% CI [{r['ci_low']*100:.1f}, "
              f"{r['ci_high']*100:.1f}]) n={r['n_seed']}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Esegui i test, devono passare**

Run: `pytest src/my_usv/scripts/test_aggregate_seeds.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/aggregate_seeds.py src/my_usv/scripts/test_aggregate_seeds.py
git commit -m "feat(analysis): aggregate_seeds — mean/std/IQM/bootstrap CI"
```

---

## Task 4: Round-robin deterministico + last_info in usv_env

**Files:**
- Modify: `src/my_usv/scripts/usv_env.py`

Questo task tocca codice ROS-coupled: la logica pura (round-robin, sector_distances) è già coperta dai test di Task 2. Qui si fa il wiring + smoke test manuale.

- [ ] **Step 1: Importa gli helper**

In `usv_env.py` riga 14, estendere l'import:
```python
from usv_logic import (
    process_lidar, compute_reward, sector_distances, round_robin_spawn,
    LIDAR_MAX_RANGE, LIDAR_BEAMS, LINEAR_VEL,
)
```

- [ ] **Step 2: Inizializza contatore round-robin e last_info**

In `__init__`, dopo `self.last_spawn = (0.0, 0.0, 0.0)` (riga 88), aggiungere:
```python
        self._test_spawn_idx = {1: 0, 2: 0, 3: 0}   # round-robin per maze (test_mode)
        self.last_info       = {                       # popolato da step_action
            'front': LIDAR_MAX_RANGE, 'left': LIDAR_MAX_RANGE,
            'right': LIDAR_MAX_RANGE, 'min_lidar': LIDAR_MAX_RANGE,
        }
```

- [ ] **Step 3: Round-robin in `reset_environment`**

Sostituire le righe 144-147:
```python
        spawn_list = TEST_SPAWN_LISTS[maze_id] if test_mode else SPAWN_LISTS[maze_id]
        for attempt in range(SPAWN_MAX_RETRIES):
            x, y, yaw = random.choice(spawn_list)
            self._teleport(x, y, yaw)
```
con:
```python
        spawn_list = TEST_SPAWN_LISTS[maze_id] if test_mode else SPAWN_LISTS[maze_id]
        if test_mode:
            chosen = round_robin_spawn(spawn_list, self._test_spawn_idx[maze_id])
            self._test_spawn_idx[maze_id] += 1
        for attempt in range(SPAWN_MAX_RETRIES):
            x, y, yaw = chosen if test_mode else random.choice(spawn_list)
            self._teleport(x, y, yaw)
```

- [ ] **Step 4: Popola `last_info` in `step_action`**

Sostituire le righe 228-229:
```python
        self._push_frame(scan_for_state)
        return self.get_state(), reward, done
```
con:
```python
        self.last_info = sector_distances(self.current_scan)
        self._push_frame(scan_for_state)
        return self.get_state(), reward, done
```

- [ ] **Step 5: Smoke test ROS (manuale)**

Avviare Gazebo su M2 e lanciare un mini-test (dopo aver completato Task 7, oppure con un check rapido in container):
```bash
# In container, con Gazebo M2 attivo:
python3 -c "
import rclpy; from usv_env import UsvEnv
rclpy.init(); e = UsvEnv()
for i in range(8):
    e.reset_environment(maze_id=2, test_mode=True)
    print(i, e.last_spawn)
"
```
Expected: la sequenza di `last_spawn` cicla sui 6 spawn di M2 e poi ricomincia (round-robin). Nessun crash. `e.last_info` valorizzato dopo uno `step_action`.

- [ ] **Step 6: Commit**

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "feat(env): round-robin spawn in test + last_info per-settore"
```

---

## Task 5: Seed nel checkpoint (train_core)

**Files:**
- Modify: `src/my_usv/scripts/train_core.py`

- [ ] **Step 1: Aggiungi `seed` a `save_ckpt`**

In `train_core.py`, cambiare la firma (riga 98) e il dict salvato. Nuova firma:
```python
def save_ckpt(agent, episode, rh, crashes, path, best_avg=-float('inf'), seed=None):
```
Aggiungere nel dict `data` (dopo `'best_avg': best_avg,`):
```python
        'seed':           seed,
```

- [ ] **Step 2: Smoke (manuale) — backward compat**

`load_ckpt` non richiede modifiche: usa `.get` per le chiavi opzionali e ignora `seed` (il seed runtime arriva sempre da CLI). Verifica rapida che il modulo importa:

Run: `python3 -c "import train_core; print('ok')"` (in container)
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add src/my_usv/scripts/train_core.py
git commit -m "feat(ckpt): traccia il seed nel checkpoint"
```

---

## Task 6: Seed + crash_sector in train.py

**Files:**
- Modify: `src/my_usv/scripts/train.py`

- [ ] **Step 1: Import e CLI**

In `train.py` riga 35-39, estendere l'import:
```python
from usv_env import UsvEnv
from usv_logic import crash_sector
from seeding import set_global_seed
from train_core import (
    DDQNAgent, save_ckpt, load_ckpt,
    EPSILON_MIN, BETA_DECAY, REPLAY_START_SIZE,
)
```
In `parse_args` (dopo `--total-ep`, riga 51) aggiungere:
```python
    p.add_argument('--seed', type=int, default=0)
```

- [ ] **Step 2: Applica il seed come prima istruzione di main**

In `main()`, subito dopo `args = parse_args()` (riga 56):
```python
    set_global_seed(args.seed)
```

- [ ] **Step 3: Header CSV con `crash_sector`**

Nella `csv_w.writerow([...])` dell'header (righe 82-86), aggiungere `'crash_sector'` in fondo:
```python
        csv_w.writerow([
            'ep_global', 'maze', 'steps', 'reward',
            'avg100', 'epsilon', 'avg_loss', 'crashed',
            'total_steps', 'total_crashes', 'spawn', 'crash_sector'
        ])
```

- [ ] **Step 4: Calcola e scrivi `crash_sector`**

Dopo il loop interno degli step, prima di scrivere la riga CSV (prima della riga 155 `csv_w.writerow([`), inserire:
```python
        crash_sec = ''
        if done:
            li = env.last_info
            crash_sec = crash_sector(li['front'], li['left'], li['right'])
```
e aggiungere `crash_sec` in coda alla riga CSV (riga 155-160):
```python
        csv_w.writerow([
            ep_disp, args.maze_id, steps + 1,
            round(ep_rew, 2), round(avg100, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            int(done), agent.total_steps, crashes, spawn_label, crash_sec
        ])
```

- [ ] **Step 5: Passa il seed a `save_ckpt`**

Sostituire le due chiamate `save_ckpt(...)` (riga 91 in `_exit`, riga 164 nel loop) aggiungendo `seed=args.seed`:
```python
        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint, best_avg, seed=args.seed)
```
```python
            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint, best_avg, seed=args.seed)
```

- [ ] **Step 6: Smoke (manuale) — import**

Run: `python3 -c "import train; print('ok')"` (in container)
Expected: `ok` (nessun errore di import/sintassi)

- [ ] **Step 7: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "feat(train): --seed + colonna crash_sector nel log"
```

---

## Task 7: Logging completo in test.py

**Files:**
- Modify (riscrive): `src/my_usv/scripts/test.py`

Riscrittura completa: nuovi CLI (`--seed`, `--reps`, `--config`, `--out-dir`, `--log-q-full`), eval round-robin (gli episodi derivano da `n_spawns × reps`), logging per-step + crash + summary.

- [ ] **Step 1: Sostituisci interamente `test.py`**

`src/my_usv/scripts/test.py`:
```python
"""test.py – Valutazione policy DDQN su un singolo maze, con logging rigoroso.

CLI:
  --maze-id     INT   1/2/3
  --model       STR   path best_model.pth
  --reps        INT   ripetizioni per spawn (episodi = n_spawn × reps; default 30)
  --episodes    INT   override opzionale del totale (se >0 ignora --reps)
  --seed        INT   seed globale (default 0)
  --config      STR   etichetta config (default 'default')
  --out-dir     STR   cartella output (default runs/default/seed_0)
  --log-q-full        se presente, logga tutti gli 11 Q-values per step

ε=0.0 (greedy). Successo = MAX_STEPS raggiunti senza collisione.
"""
import argparse
import csv
import os
import sys
from collections import deque

import numpy as np
import rclpy
import torch

from ddqn_model import DDQN
from usv_env import UsvEnv, TEST_SPAWN_LISTS
from usv_logic import sector_distances, crash_sector
from seeding import set_global_seed

MAX_STEPS = 500


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--maze-id',     type=int, default=1)
    p.add_argument('--model',       type=str,
                   default='src/my_usv/scripts/best_ddqn_model.pth')
    p.add_argument('--reps',        type=int, default=30)
    p.add_argument('--episodes',    type=int, default=0)
    p.add_argument('--seed',        type=int, default=0)
    p.add_argument('--config',      type=str, default='default')
    p.add_argument('--out-dir',     type=str, default='runs/default/seed_0')
    p.add_argument('--log-q-full',  action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)

    if not os.path.exists(args.model):
        print(f"[ERRORE] Modello non trovato: {args.model}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    m = args.maze_id
    steps_path   = os.path.join(args.out_dir, f'eval_steps_m{m}.csv')
    crashes_path = os.path.join(args.out_dir, f'eval_crashes_m{m}.csv')
    summary_path = os.path.join(args.out_dir, 'eval_summary.csv')

    n_spawns = len(TEST_SPAWN_LISTS[m])
    episodes = args.episodes if args.episodes > 0 else n_spawns * args.reps

    q_net = DDQN()
    q_net.load_state_dict(torch.load(args.model, map_location='cpu'))
    q_net.eval()
    print(f"  ✅ Modello: {args.model} | Maze {m} | {episodes} ep "
          f"({n_spawns} spawn × {args.reps} reps) | seed {args.seed}")

    # ── CSV per-step ──────────────────────────────────────────────
    steps_header = ['episode', 'step', 'spawn', 'action',
                    'q_chosen', 'q_max', 'q_spread',
                    'front_dist', 'left_dist', 'right_dist', 'min_lidar',
                    'reward', 'done']
    if args.log_q_full:
        steps_header += [f'q{i}' for i in range(11)]
    steps_f = open(steps_path, 'w', newline='', encoding='utf-8')
    steps_w = csv.writer(steps_f)
    steps_w.writerow(steps_header)

    # ── CSV crash ─────────────────────────────────────────────────
    crashes_f = open(crashes_path, 'w', newline='', encoding='utf-8')
    crashes_w = csv.writer(crashes_f)
    crashes_w.writerow(['episode', 'spawn', 'crash_step',
                        'crash_sector', 'crash_dist', 'last_actions'])

    rclpy.init()
    env = UsvEnv()

    rewards, steps_l, crashes = [], [], 0
    spawn_stats = {}

    for ep in range(1, episodes + 1):
        state = env.reset_environment(maze_id=m, test_mode=True)
        sx, sy, _ = env.last_spawn
        spawn_label = f"({sx:.1f},{sy:.1f})"
        spawn_stats.setdefault(spawn_label, {'total': 0, 'completed': 0})
        ep_reward = 0.0
        ep_steps = 0
        crashed = False
        recent_actions = deque(maxlen=5)

        for step in range(MAX_STEPS):
            with torch.no_grad():
                q = q_net(torch.FloatTensor(state).unsqueeze(0)).squeeze(0)
            action = int(q.argmax().item())
            q_chosen = float(q[action])
            q_max = float(q.max())
            q_spread = q_max - float(q.min())
            recent_actions.append(action)

            state, reward, done = env.step_action(action, training=False)
            sd = env.last_info
            ep_reward += reward
            ep_steps += 1

            row = [ep, step, spawn_label, action,
                   round(q_chosen, 4), round(q_max, 4), round(q_spread, 4),
                   round(sd['front'], 4), round(sd['left'], 4),
                   round(sd['right'], 4), round(sd['min_lidar'], 4),
                   round(reward, 4), int(done)]
            if args.log_q_full:
                row += [round(float(x), 4) for x in q.tolist()]
            steps_w.writerow(row)

            if done:
                crashed = True
                crashes += 1
                sec = crash_sector(sd['front'], sd['left'], sd['right'])
                crashes_w.writerow([
                    ep, spawn_label, step, sec,
                    round(sd['min_lidar'], 4),
                    ','.join(str(a) for a in recent_actions)])
                break

        steps_f.flush()
        crashes_f.flush()
        rewards.append(ep_reward)
        steps_l.append(ep_steps)
        spawn_stats[spawn_label]['total'] += 1
        if not crashed:
            spawn_stats[spawn_label]['completed'] += 1

        esito = '💥 CRASH' if crashed else '✅ OK   '
        print(f"  Ep {ep:>4}/{episodes}  {ep_steps:>4} step  "
              f"R:{ep_reward:>8.1f}  {spawn_label:<14} {esito}")

    n_success = episodes - crashes
    success_rate = n_success / episodes
    avg_reward = float(np.mean(rewards))
    avg_steps = float(np.mean(steps_l))

    # ── eval_summary.csv (append: una riga per maze) ──────────────
    summary_is_new = (not os.path.exists(summary_path) or
                      os.path.getsize(summary_path) == 0)
    with open(summary_path, 'a', newline='', encoding='utf-8') as sf:
        sw = csv.writer(sf)
        if summary_is_new:
            sw.writerow(['config', 'seed', 'maze', 'episodes',
                         'n_success', 'success_rate', 'avg_reward', 'avg_steps'])
        sw.writerow([args.config, args.seed, m, episodes, n_success,
                     round(success_rate, 4), round(avg_reward, 2),
                     round(avg_steps, 1)])

    print(f"\n  📊 Maze {m}: success {success_rate*100:.1f}% "
          f"({n_success}/{episodes}) | reward {avg_reward:.1f} | "
          f"steps {avg_steps:.1f}")
    print("  Per-spawn:")
    for sp, s in sorted(spawn_stats.items(),
                        key=lambda x: -x[1]['completed'] / max(x[1]['total'], 1)):
        rate = s['completed'] / s['total'] * 100
        print(f"    {sp:<14} {s['completed']}/{s['total']}  ({rate:.1f}%)")

    steps_f.close()
    crashes_f.close()
    env.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke (manuale) — import**

Run: `python3 -c "import test; print('ok')"` (in container)
Expected: `ok`

- [ ] **Step 3: Smoke (manuale) — mini eval su M2**

Con Gazebo M2 attivo, modello presente in `--out-dir`:
```bash
python3 src/my_usv/scripts/test.py --maze-id 2 --reps 1 --seed 0 \
  --model runs/default/seed_0/best_model.pth --out-dir runs/default/seed_0
```
Expected: 6 episodi (6 spawn × 1). I file `eval_steps_m2.csv`, `eval_crashes_m2.csv`, `eval_summary.csv` esistono con le colonne attese. `eval_steps_m2.csv` ha una riga per step con `q_chosen/front_dist/...`.

- [ ] **Step 4: Commit**

```bash
git add src/my_usv/scripts/test.py
git commit -m "feat(test): eval round-robin + logging per-step/crash/summary"
```

---

## Task 8: Orchestrazione — start scripts, run_meta, .gitignore

**Files:**
- Modify: `start_test.sh`, `start_train_multimaze.sh`, `.gitignore`

I file shell si verificano con `bash -n` (parse) + esecuzione reale durante una run.

- [ ] **Step 1: `.gitignore` ignora `runs/`**

Aggiungere in fondo a `.gitignore` (crearlo se assente):
```
# Artefatti di training/eval pesanti (checkpoint, CSV per-step). Solo gli
# aggregati in ANALISI_TRAINING/ vanno versionati.
runs/
```

- [ ] **Step 2: `start_train_multimaze.sh` — seed/config + layout runs/ + backup-guard**

Dopo la riga 20 (`BLOCK_PATTERN=(1 2 2) ...`), aggiungere parsing args:
```bash
SEED=0
CONFIG="default"
DO_RESET=0
for arg in "$@"; do
    case "$arg" in
        --reset)        DO_RESET=1 ;;
        --seed=*)       SEED="${arg#*=}" ;;
        --config=*)     CONFIG="${arg#*=}" ;;
    esac
done
RUN_DIR="runs/${CONFIG}/seed_${SEED}"
mkdir -p "$(pwd)/${RUN_DIR}"
```

Sostituire la definizione del checkpoint (riga 28):
```bash
CHECKPOINT_CTR="/home/usv_ws/${RUN_DIR}/checkpoint.pkl"
```

Sostituire il blocco `--reset` (righe 34-40) con backup-guard:
```bash
if [[ "$DO_RESET" == "1" ]]; then
    BACKUP_DIR="ANALISI_TRAINING/$(date +%Y_%m_%d)/pre_reset_${CONFIG}_seed_${SEED}"
    if [[ -d "${RUN_DIR}" ]]; then
        echo "  --reset: backup di ${RUN_DIR} → ${BACKUP_DIR}"
        mkdir -p "${BACKUP_DIR}"
        if ! cp -r "${RUN_DIR}/." "${BACKUP_DIR}/"; then
            echo "  ❌ Backup fallito. Reset abortito per non perdere dati."
            exit 1
        fi
    fi
    echo "  --reset: rimozione artefatti in ${RUN_DIR}..."
    rm -f "${RUN_DIR}/checkpoint.pkl" "${RUN_DIR}/training_log.csv" \
          "${RUN_DIR}/best_ddqn_model.pth" "${RUN_DIR}/best_model.pth"
    echo "  Reset completato."
fi
```

Aggiungere `--seed` alla chiamata `train.py` (dentro `docker exec`, dopo `--checkpoint ${CHECKPOINT_CTR}`, riga 118):
```bash
                --checkpoint ${CHECKPOINT_CTR} \
                --seed       ${SEED}
```

- [ ] **Step 3: `start_test.sh` — seed/config/reps + layout runs/ + run_meta.json**

Sostituire il blocco CONFIGURAZIONE (righe 22-32) con:
```bash
SEED=0
CONFIG="default"
REPS=30
for arg in "$@"; do
    case "$arg" in
        --seed=*)    SEED="${arg#*=}" ;;
        --config=*)  CONFIG="${arg#*=}" ;;
        --reps=*)    REPS="${arg#*=}" ;;
    esac
done

GAZEBO_SPEED=3
GAZEBO_WAIT=25
RUN_DIR="runs/${CONFIG}/seed_${SEED}"
mkdir -p "$(pwd)/${RUN_DIR}"

MODEL_PATH="${RUN_DIR}/best_model.pth"
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
MODEL_CTR="/home/usv_ws/${RUN_DIR}/best_model.pth"
OUT_DIR_CTR="/home/usv_ws/${RUN_DIR}"
PATCHED_WORLD="/tmp/world_fast.world"

# run_meta.json: provenienza (git SHA, host, config, seed)
cat > "$(pwd)/${RUN_DIR}/run_meta.json" <<META
{
  "config": "${CONFIG}",
  "seed": ${SEED},
  "reps": ${REPS},
  "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "hostname": "$(hostname)",
  "timestamp": "$(date -Iseconds)",
  "success_criterion": "MAX_STEPS=500 reached without collision, epsilon=0.0"
}
META
```

Sostituire la pulizia CSV (riga 70 `rm -f "$OUTPUT_CSV"`) con:
```bash
rm -f "${RUN_DIR}/eval_summary.csv"
```

Sostituire `run_test()` (righe 144-158) con:
```bash
run_test() {
    local maze_id=$1
    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/test.py \
                --maze-id ${maze_id} \
                --model   ${MODEL_CTR} \
                --reps    ${REPS} \
                --seed    ${SEED} \
                --config  ${CONFIG} \
                --out-dir ${OUT_DIR_CTR}
        "
    return $?
}
```

Il report finale awk (righe 195-290) legge `$OUTPUT_CSV`: aggiornare la variabile a livello di report sostituendo `OUTPUT_CSV` con `${RUN_DIR}/eval_summary.csv` **non** è corretto (summary ha schema diverso). Per ora il report comparativo dettagliato è sostituito da `aggregate_seeds.py` (Task 3). Sostituire l'intero blocco da riga 185 alla fine con:
```bash
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Eval completata. Output in ${RUN_DIR}/"
echo "  eval_summary.csv  eval_steps_m*.csv  eval_crashes_m*.csv"
echo ""
echo "  Per aggregare più seed:"
echo "    python3 src/my_usv/scripts/aggregate_seeds.py \\"
echo "      --config ${CONFIG} --output ANALISI_TRAINING/\$(date +%Y_%m_%d)/aggregate_${CONFIG}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
```

Aggiornare anche il check modello (riga 62) e l'header stampato: il check `if [ ! -f "$MODEL_PATH" ]` resta valido (ora punta a `${RUN_DIR}/best_model.pth`).

- [ ] **Step 4: Verifica sintassi shell**

Run: `bash -n start_train_multimaze.sh && bash -n start_test.sh && echo OK`
Expected: `OK` (nessun errore di parse)

- [ ] **Step 5: Commit**

```bash
git add start_train_multimaze.sh start_test.sh .gitignore
git commit -m "feat(orchestration): --seed/--config, layout runs/, backup-guard, run_meta"
```

---

## Task 9: Verifica end-to-end (1 seed)

**Files:** nessuna modifica — validazione integrata.

- [ ] **Step 1: Test suite completa**

Run (in container): `pytest src/my_usv/scripts/test_seeding.py src/my_usv/scripts/test_logic_helpers.py src/my_usv/scripts/test_aggregate_seeds.py -v`
Expected: PASS (11 passed)

- [ ] **Step 2: Mini run di training seedata**

```bash
./start_train_multimaze.sh --seed=0 --config=smoke
```
Interrompere dopo ~1 blocco. Verificare che esistano `runs/smoke/seed_0/checkpoint.pkl` e `runs/smoke/seed_0/training_log.csv`, e che il CSV abbia la colonna `crash_sector`.

- [ ] **Step 3: Eval seedata**

Copiare il `best_ddqn_model.pth` prodotto in `runs/smoke/seed_0/best_model.pth`, poi:
```bash
./start_test.sh --seed=0 --config=smoke --reps=2
```
Expected: `runs/smoke/seed_0/` contiene `eval_summary.csv`, `eval_steps_m{1,2,3}.csv`, `eval_crashes_m{1,2,3}.csv`, `run_meta.json`.

- [ ] **Step 4: Aggregazione**

```bash
python3 src/my_usv/scripts/aggregate_seeds.py --config smoke \
  --output ANALISI_TRAINING/smoke_aggregate.csv
```
Expected: stampa per-maze `mean ± std (IQM; 95% CI)` e scrive il CSV aggregato (n_seed=1).

- [ ] **Step 5: Riproducibilità**

Rilanciare lo stesso eval con `--seed=0`: la sequenza di spawn nei CSV per-step è identica (round-robin deterministico). NB: i valori fisici possono variare leggermente (Gazebo non deterministico) — atteso e documentato.

- [ ] **Step 6: Commit (se servono fix)**

```bash
git add -A
git commit -m "test: verifica end-to-end base rigorosa (1 seed)"
```

---

## Self-Review (eseguita)

**Spec coverage:**
- Pilastro 1 (seed) → Task 1, 5, 6, 7 (CLI + persistenza). ✅
- Pilastro 2 (eval round-robin + criterio) → Task 4 (round-robin), Task 7 (reps, summary, criterio). ✅
- Pilastro 3 (logging per-step + crash) → Task 7; training crash_sector → Task 6. ✅
- `last_info` interface → Task 4. ✅
- Costanti settore condivise → Task 2. ✅
- Layout `runs/<config>/seed_<S>/` → Task 8. ✅
- Backup-guard + `.gitignore` + `run_meta.json` → Task 8. ✅
- Aggregazione mean/std/IQM/bootstrap CI → Task 3. ✅
- `--log-q-full` → Task 7. ✅

**Type consistency:** `sector_distances` ritorna dict con chiavi `right/front/left/min_lidar`, usato coerentemente in Task 4/6/7. `crash_sector(front,left,right)` firma coerente in Task 2/6/7. `eval_summary.csv` schema `config,seed,maze,episodes,n_success,success_rate,avg_reward,avg_steps` coerente tra Task 7 (scrittura) e Task 3 (lettura). ✅

**Placeholder scan:** nessun TBD/TODO; tutti gli step hanno codice/comandi concreti. ✅

**Nota:** lo smoke test ROS di Task 4 Step 5 dipende da Gazebo attivo; in subagent-driven può essere marcato come verifica manuale dall'utente (richiede Docker+Gazebo). I task pure-Python (1,2,3) sono pienamente automatizzabili.

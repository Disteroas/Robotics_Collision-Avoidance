# Design: paper_implementation branch

**Date**: 2026-05-07  
**Branch**: `paper_implementation` (da `curriculum_learning`)  
**Obiettivo**: eliminare le 4 root cause del fallimento di `curriculum_learning` e implementare il training secondo Feng et al. 2021.

---

## Root cause fallimento curriculum_learning

1. **Spawn fisso** → memorizzazione percorso, no generalizzazione
2. **β=0.995** → ε ≈ 0.08 quando Phase 2 inizia (~ep 600), esplorazione insufficiente su Maze 2
3. **Reward complessa** (space_bonus + front_danger 3.0m + steering) → segnali fuorvianti in Maze 2 (81% muri diagonali)
4. **Distributional shift** → Maze 1 axis-aligned, Maze 2 pattern LIDAR completamente diversi

---

## Architettura

### Branch
```
paper_implementation  ←  fork da curriculum_learning
```

### File modificati
| File | Modifica |
|---|---|
| `src/my_usv/scripts/usv_logic.py` | Reward semplificata |
| `src/my_usv/scripts/train_core.py` | `BETA_DECAY` 0.995 → 0.999 |
| `src/my_usv/scripts/train.py` | ε reset Phase 2 + `maze_id` a `env.reset()` |
| `src/my_usv/scripts/usv_env.py` | Spawn list + `_teleport()` via `SetEntityState` |
| `start_training_curriculum.sh` | Rimuove gestione spawn (ora in `usv_env.py`) |

### File aggiunti (già creati)
| File | Scopo |
|---|---|
| `src/my_usv/scripts/validate_spawn.py` | Validatore spawn point (exit 0/1/2/3) |
| `test_spawns.sh` | Testa tutti gli spawn point prima del training |

---

## 1. Reward

**Feng et al. 2021 — pura**:
```python
def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False
```

Rimosse da `usv_logic.py`: `FRONT_DANGER`, `SIDE_DANGER`, `SPACE_BONUS_WEIGHT`,
`space_bonus`, `steering_penalty`, `danger_penalty`.  
`action_index` mantenuto nella firma per retrocompatibilità con `usv_env.py`.

---

## 2. ε decay policy

### BETA_DECAY
```python
BETA_DECAY = 0.999   # era 0.995
```

Progressione con β=0.999, ε₀=1.0:
```
ep  100 → ε=0.905    ep  600 → ε=0.549    ep 2000 → ε=0.135
ep  300 → ε=0.741    ep 1000 → ε=0.368    ep 3000 → ε=0.050  ← minimo esatto
```
Curva completa in 3000 episodi senza troncare. GAMMA=0.99 invariato.

### Reset ε a Phase 2 (Narvekar et al. 2020 — task-boundary exploration reset)
```python
EPSILON_RESET_P2 = 0.5

# in train.py, quando phase 1→2 si sblocca:
agent.epsilon = max(agent.epsilon, EPSILON_RESET_P2)
```
`max()` garantisce: se ε già > 0.5 (transizione precoce) non si abbassa; se ε < 0.5 (transizione tardiva, es. ep > 693) si porta a 0.5.  
Il valore viene salvato nel checkpoint → blocchi successivi partono con ε corretto.

---

## 3. Random spawn per-episode (Opzione B)

### Meccanismo
A ogni `reset_environment(maze_id)`:
1. `/reset_world` — pulisce fisica Gazebo, resetta a pose di lancio
2. `_teleport(x, y, yaw)` — sposta robot a spawn random scelto da lista
3. Drain LIDAR 20 cicli (invariato)
4. `_wait_sim_seconds(0.8)` (invariato)

### Implementazione teleport
```python
# usv_env.py — nuovo metodo
from gazebo_msgs.srv import SetEntityState
import math, random

SPAWN_LISTS = {
    1: [
        (-3.0, -5.0,  1.57),  # M1-A1: south open, heading N
        ( 0.0, -4.5,  1.57),  # M1-A2: centre-south, heading N
        ( 2.5, -5.0,  1.57),  # M1-A3: right-south, heading N
        (-1.5, -5.0,  0.0 ),  # M1-A4: south open, heading E
        (-2.9, -2.0,  1.57),  # M1-B1: left channel entry, heading N
        (-2.9,  0.5,  0.0 ),  # M1-B2: left channel mid, heading E
        ( 2.5, -2.0,  1.57),  # M1-C1: right outer, heading N
        ( 0.5, -2.5,  1.57),  # M1-D1: centre-bottom, heading N
    ],
    2: [
        (-6.0,  0.0,  0.0 ),  # M2-A1: left entrance, heading E
        (-6.0, -1.5,  0.0 ),  # M2-A2: lower-left, heading E
        (-6.0,  2.0,  0.0 ),  # M2-A3: upper-left, heading E
        (-6.0,  0.0,  1.57),  # M2-A4: left entrance, heading N
        (-3.5,  0.5,  0.0 ),  # M2-B1: centre-left, heading E
        (-3.5, -2.5,  1.57),  # M2-B2: centre-left low, heading N
        (-1.5, -2.5,  0.0 ),  # M2-C1: between Wall_32/Wall_20, heading E
        ( 1.5,  0.0,  3.14),  # M2-D1: right-centre, heading W
    ],
}

def _teleport(self, x: float, y: float, yaw: float) -> None:
    req = SetEntityState.Request()
    req.state.name = 'usv_robot'
    req.state.pose.position.x = float(x)
    req.state.pose.position.y = float(y)
    req.state.pose.position.z = 0.0
    req.state.pose.orientation.x = 0.0
    req.state.pose.orientation.y = 0.0
    req.state.pose.orientation.z = math.sin(yaw / 2.0)
    req.state.pose.orientation.w = math.cos(yaw / 2.0)
    req.state.twist.linear.x  = 0.0
    req.state.twist.angular.z = 0.0
    future = self.teleport_client.call_async(req)
    while not future.done():
        rclpy.spin_once(self, timeout_sec=0.01)
    self._wait_sim_seconds(0.3)
```

`_teleport()` chiamato DOPO `/reset_world` in `reset_environment(maze_id)`.

### Spawn validation
Prima del training → eseguire `./test_spawns.sh` e rimuovere eventuali `❌ COLLISION`.

---

## 4. Curriculum (struttura invariata + ε reset)

```
Phase 1: solo Maze 1
  - spawn random da SPAWN_LISTS[1] ogni episodio
  - threshold: avg50_maze1 > 1500 reward → sblocca Phase 2
  - al passaggio 1→2: agent.epsilon = max(agent.epsilon, 0.5)

Phase 2: 30% Maze 1 + 70% Maze 2   (PHASE2_PROB=70, invariato)
  - spawn random da SPAWN_LISTS[maze_corrente] ogni episodio
```

---

## 5. Parametri — riepilogo

| Parametro | `curriculum_learning` | `paper_implementation` |
|---|---|---|
| `BETA_DECAY` | 0.995 | **0.999** |
| ε reset Phase 2 | — | **max(ε, 0.5)** |
| Reward | +5+bonus−penalties / −1000 | **+5 / −1000** |
| Spawn | fisso per maze | **random per-episode da lista** |
| `GAMMA` | 0.99 | 0.99 |
| `PHASE2_THRESHOLD` | 1500 | 1500 |
| `MAX_STEPS` | 1000 | 1000 |
| `MEMORY_CAPACITY` | 100,000 | 100,000 |
| Episodes totali | 3000 | 3000 |
| Gazebo speed | 4× | 4× |
| Phase 2 split | 30/70 | 30/70 |

---

## 6. Invariato

- Architettura rete: 50→300→300→11 (ReLU, fully connected)
- Adam LR=0.00025, batch=64, target update ogni 1000 step
- Formato checkpoint (retrocompatibile)
- `test.py` — valutazione greedy su Maze 1/2/3
- Infrastruttura Docker, `start_training_curriculum.sh` (solo rimozione gestione spawn)

---

## Riferimenti letteratura

- **Feng et al. 2021** — reward +step / −collision, architettura DDQN base
- **Narvekar et al. 2020** — curriculum RL, task-boundary exploration reset
- **Bengio et al. 2009** — curriculum learning, progressive difficulty

# Sessione implementazione: branch paper_implementation

**Data:** 2026-05-07  
**Branch:** `paper_implementation` (fork da `curriculum_learning`)  
**Obiettivo:** Eliminare le 4 root cause del fallimento di `curriculum_learning` e implementare il training secondo Feng et al. 2021.

---

## Contesto: perché è fallito curriculum_learning

Training precedente (`curriculum_learning`, 3000 ep) aveva prodotto risultati:
- Maze 1: 100% successo
- Maze 2: 0% successo
- Maze 3: 0% successo

Diagnosi root cause:

| # | Problema | Effetto |
|---|---|---|
| 1 | **Spawn fisso** | Robot memorizza percorso specifico, non generalizza |
| 2 | **β=0.995** | ε≈0.08 quando Phase 2 inizia (~ep 600) — esplorazione insufficiente su Maze 2 |
| 3 | **Reward complessa** (space_bonus + front_danger 3.0m + steering) | Segnali fuorvianti in Maze 2 (81% muri diagonali) |
| 4 | **Distributional shift** | Maze 1 solo axis-aligned; Maze 2 pattern LIDAR completamente diversi |

---

## Modifiche implementate

### 1. Reward semplificata — `usv_logic.py`

**Riferimento:** Feng et al. 2021

Rimosse: costanti `FRONT_DANGER`, `SIDE_DANGER`, `SPACE_BONUS_WEIGHT`, logiche `space_bonus`, `steering_penalty`, `danger_penalty`.

```python
def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False
```

`action_index` mantenuto nella firma per retrocompatibilità con `usv_env.py`.

### 2. β=0.999 — `train_core.py`

```python
BETA_DECAY = 0.999   # era 0.995
```

Progressione ε con β=0.999, ε₀=1.0:

| Episodio | ε |
|---|---|
| 100 | 0.905 |
| 300 | 0.741 |
| 600 | 0.549 |
| 1000 | 0.368 |
| 2000 | 0.135 |
| 3000 | 0.050 ← minimo esatto |

Con β=0.995 (vecchio), ε=0.05 veniva raggiunto a ep ~600 — troppo presto, Phase 2 iniziava con esplorazione quasi nulla.

### 3. ε reset a Phase 2 — `train.py`

**Riferimento:** Narvekar et al. 2020 (task-boundary exploration reset)

```python
EPSILON_RESET_P2 = 0.5

# quando phase 1→2 si sblocca:
agent.epsilon = max(agent.epsilon, EPSILON_RESET_P2)
```

`max()` garantisce:
- Se ε > 0.5 (transizione precoce): non si abbassa
- Se ε < 0.5 (transizione tardiva, es. ep > 693): si porta a 0.5

Inoltre `env.reset_environment()` → `env.reset_environment(maze_id=args.maze_id)` per passare il maze corrente all'ambiente.

### 4. Spawn random per-episode — `usv_env.py`

A ogni `reset_environment(maze_id)`:
1. `/reset_world` — resetta fisica Gazebo
2. `_teleport(x, y, yaw)` — sposta robot a spawn casuale da lista

```python
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
```

Implementazione teleport via `gazebo_msgs/srv/SetEntityState`:

```python
def _teleport(self, x: float, y: float, yaw: float) -> None:
    req = SetEntityState.Request()
    req.state.name = 'usv_robot'
    req.state.pose.position.x = float(x)
    req.state.pose.position.y = float(y)
    req.state.pose.position.z = 0.0
    req.state.pose.orientation.z = math.sin(yaw / 2.0)  # quaternion
    req.state.pose.orientation.w = math.cos(yaw / 2.0)
    req.state.twist.linear.x  = 0.0
    req.state.twist.angular.z = 0.0
    future = self.teleport_client.call_async(req)
    while not future.done():
        rclpy.spin_once(self, timeout_sec=0.01)
    self._wait_sim_seconds(0.3)
```

### 5. Script validazione spawn — `test_spawns.sh` + `validate_spawn.py`

`validate_spawn.py`: gira dentro container Docker, legge primo scan LIDAR, controlla distanza minima da muri.
- Exit 0 = OK (min > 0.40m)
- Exit 1 = COLLISION (min < 0.25m) → rimuovere
- Exit 2 = TIMEOUT
- Exit 3 = WARNING (0.25–0.40m)

`test_spawns.sh`: cicla su tutti gli spawn point, avvia Gazebo headless per ognuno, riporta tabella risultati.

```bash
./test_spawns.sh 1   # testa Maze 1 (~7 min)
./test_spawns.sh 2   # testa Maze 2 (~7 min)
```

---

## Parametri confronto

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
| Phase 2 split | 30/70 | 30/70 |

---

## Commit history

```
5c1dd45 feat(epsilon): reset to max(eps, 0.5) on Phase 2 + pass maze_id to env.reset()
a5cefef docs(train-script): clarify SPAWN[] is launch-only; per-episode spawn in usv_env.py
f3278eb feat(spawn): per-episode random spawn via SetEntityState teleport
8aa6b61 feat(epsilon): BETA_DECAY 0.995 → 0.999 for full 3000-ep decay curve
53e513a feat(reward): simplify to +5/-1000 per Feng et al. 2021
5d19d5e chore: init paper_implementation branch from curriculum_learning
```

---

## Test suite

**34/34 test passati** su `paper_implementation`:
- `test_agent.py` — 9 test (include verifica BETA_DECAY applicato per episodio)
- `test_ddqn_model.py` — 6 test
- `test_replay_buffer.py` — 6 test
- `test_usv_logic.py` — 13 test (7 process_lidar + 6 compute_reward semplificata)

**AST check:** `train.py`, `usv_env.py`, `usv_logic.py`, `train_core.py` — tutti puliti.

---

## Prossimo step

**Prima del training** — validare spawn point (richiede Docker + Gazebo attivi):
```bash
./test_spawns.sh 1
./test_spawns.sh 2
```
Rimuovere da `SPAWN_LISTS` in `usv_env.py` e da `test_spawns.sh` eventuali spawn `❌ COLLISION`.

**Avvio training:**
```bash
./start_training_curriculum.sh
```

**Monitoring:** log in `logs/block_N_maze_M.log`. Checkpoint ogni 100 ep in `src/my_usv/scripts/checkpoint.pkl`.

---

## Riferimenti

- **Feng et al. 2021** — reward +step/−collision, architettura DDQN base per collision avoidance USV
- **Narvekar et al. 2020** — curriculum RL, task-boundary exploration reset
- **Bengio et al. 2009** — curriculum learning, progressive difficulty

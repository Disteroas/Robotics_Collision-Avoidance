# Architettura del sistema — USV DDQN Collision Avoidance

## Cos'è

Un agente DDQN (Double Deep Q-Network) che impara a navigare un USV (unmanned surface vehicle) in labirinti simulati evitando collisioni. La simulazione gira in Gazebo (ROS 2) dentro un container Docker su Windows.

Il task nella versione attuale è **collision avoidance pura**: l'agente non ha una destinazione, deve sopravvivere il più a lungo possibile senza toccare muri. Vedere [NEXT_STEPS.md](NEXT_STEPS.md) per l'aggiunta del goal.

---

## Stack tecnologico

```
Windows 11 + WSL2 + Ubuntu
└── Docker Desktop (WSL2 backend)
    └── Container usv_rl_project
        ├── ROS 2 Humble
        ├── Gazebo 11
        └── Python 3 + PyTorch CPU
```

File system condiviso via volume bind (`/$(pwd):/home/usv_ws`). Modifiche Python su Windows sono live nel container senza rebuild.

---

## Ambienti (Maze)

| ID | World file | Struttura | Ruolo |
|----|-----------|-----------|-------|
| 1 | `labirinto_9a.world` | Corridoi ortogonali | Training (curriculum) / Test |
| 2 | `labirinto_9b.world` | Muri diagonali, ~15×13m | Training principale (feng_direct) |
| 3 | `labirinto_10.world` | Struttura diversa | Test-only, mai visto in training |

---

## Componenti core

### `usv_env.py` — ambiente RL
- Nodo ROS 2. Gestisce Gazebo reset, teleport robot, raccolta LIDAR, pubblicazione comandi.
- `reset_environment(maze_id)`: reset world → teleport a spawn casuale → safety check LIDAR → ritorna stato iniziale.
- `step_action(action_index)`: pubblica Twist → aspetta 0.1s sim-time → legge LIDAR → calcola reward.
- Spawn diversity: 8 posizioni (Maze 1), 16 posizioni in 6 zone (Maze 2). Validate con `test_spawns.sh`.
- Safety check: se `min_lidar < 0.40m` dopo teleport, ritenta (max 3 volte).

### `usv_logic.py` — stato e reward
```python
LIDAR_BEAMS     = 50     # bin output
LIDAR_MAX_RANGE = 5.0    # m
COLLISION_DIST  = 0.25   # m — soglia crash
LINEAR_VEL      = 0.5    # m/s — velocità fissa
```

**Stato:** 512 raggi LIDAR raw → min-pooled in 50 bin → divisi per 5.0 → range [0, 1].

**Reward:**
```
R = +5.0          se min_lidar >= 0.25m   (vivo)
R = -1000.0       se min_lidar <  0.25m   (crash, fine episodio)
```
Reward binario — assenza di shaping graduato. Vedere [DECISIONI.md](DECISIONI.md).

**Azioni:** 11 valori discreti di velocità angolare.
```
action_index 0..10 → angular_z = -0.8 + 0.16 * index
             → range [-0.8, +0.8] rad/s
```

### `ddqn_model.py` — rete neurale
```
Input  → 50 (LIDAR normalizzato)
FC1    → 300  (ReLU)
FC2    → 300  (ReLU)
Output → 11   (Q-values, una per azione)
```
Nessun softmax in output. Pesi inizializzati con default PyTorch (Kaiming uniform).

### `train_core.py` — agente DDQN
```python
GAMMA               = 0.99       # fattore di sconto (orizzonte ~100 step)
LR                  = 0.00025    # Adam learning rate
MEMORY_CAPACITY     = 100_000    # replay buffer
BATCH_SIZE          = 64
BETA_DECAY          = 0.999      # ε *= 0.999 per episodio
EPSILON_MIN         = 0.05       # ε raggiunge min a ~ep 3000
TARGET_UPDATE_STEPS = 1_000      # aggiorna target net ogni N step globali
```

**Training step:**
1. `agent.act(state)` → ε-greedy sull'online network
2. `env.step_action(a)` → (next_state, reward, done)
3. `memory.push(...)` → buffer circolare 100k
4. `agent.learn()` → campiona batch → calcola target DDQN → MSE loss → Adam → clip grad a 10.0

**Target DDQN:**
```
a* = argmax_a Q_online(s', a)           # online net sceglie l'azione
Q_target = r + γ · Q_target(s', a*)    # target net valuta il valore
```

### `train.py` — loop principale
- MAX_STEPS = 1000 per episodio (training)
- Salva `best_ddqn_model.pth` quando avg100 migliora
- Log CSV: `training_log.csv` (ep, maze, steps, reward, avg100, epsilon, loss, crashed, total_steps)

### `test.py` — valutazione policy
- ε = 0.0 (greedy pura)
- MAX_STEPS = 500 per episodio
- Carica `best_ddqn_model.pth`
- Output CSV: `test_results.csv`

---

## Flusso dati

```
Gazebo → /scan (LaserScan 512 ray) → process_lidar() → 50 bin
                                                          ↓
                                               get_state() → [0,1]^50
                                                          ↓
                                               DDQN.forward() → Q[11]
                                                          ↓
                                               argmax → action_index
                                                          ↓
                                               step_action() → /cmd_vel (Twist)
                                                          ↓
                                               Gazebo → nuova posizione → /scan
```

---

## File di stato runtime

| File | Contenuto | Dimensione tipica |
|------|-----------|-------------------|
| `checkpoint.pkl` | Q-net + target + optimizer + buffer + epsilon + storia | ~40 MB |
| `best_ddqn_model.pth` | Solo Q-net weights (best avg100) | ~1 MB |
| `training_log.csv` | Una riga per episodio | ~200 KB per 3000 ep |
| `test_results.csv` | 90 righe (30 ep × 3 maze) | ~5 KB |

# DDQN Round 1 — Multi-maze + Domain Randomization + D1 Relocation

**Data:** 2026-05-19
**Branch design:** `ddqn_round1_19_05` (da `ddqn_enhanced_18_05`)
**Autori:** Davide Covolo (+ Claude)

---

## 1. Obiettivo

Risolvere asintoto generalization osservato su 10 esperimenti consecutivi: M3 zero-shot = 0% costante, M1 = 0% dopo training M2-only su `ddqn_enhanced_18_05` (M2=51% record storico).

**Ipotesi falsificabile:** training su multi-environment + perception randomization permette policy generalization a maze sconosciuto (M3) mantenendo performance su training set (M2 ≥ 45%).

---

## 2. Cambiamenti rispetto a baseline `ddqn_enhanced_18_05`

| Componente | Baseline ddqn_enh | Round 1 | Razionale |
|---|---|---|---|
| Maze training | M2 only | M1 + M2 ratio 1:2 | Cobbe 2019 — multi-env per generalization |
| LIDAR noise | nessuno | σ=0.02 (training-only) | Tobin 2017, Peng 2018 — sim-to-real robustness |
| Spawn D1 M2 | (1.5, 0.0, π) | (3.5, -0.5, π/2) | Analisi geometrica — D1 originale heading W → muro <60 step |
| Frame stacking | k=3 | k=3 | invariato |
| Heading [cos,sin] | 2 dim | 2 dim | invariato (heading×10 riservato a Round 2) |
| STATE_DIM | 152 | 152 | invariato |
| Reward shaping | merge16 | merge16 | invariato |
| TARGET_UPDATE | 5000 | 5000 | invariato |
| Hyperparameters DDQN | merge16 | merge16 | tutti invariati |
| Episodi | 5000 | 5000 | invariato per confronto |

**Variabili modificate:** 3 (maze mix, DR, spawn D1).

---

## 3. Architettura

### 3.1 State vector (invariato)

```
STATE_DIM = 152
= LIDAR_BEAMS (50) × FRAME_STACK (3) = 150
+ [cos(yaw), sin(yaw)] from /odom    =   2
```

### 3.2 Reward function (invariato — `usv_logic.py` merge16)

```
+5.0 base step reward
+ SPACE_BONUS_WEIGHT * mean(scan) / LIDAR_MAX_RANGE      # max +2.0
- 0.02 * abs(action - 5)                                 # steering, max -0.10
- 20 * severity²  if front_dist < FRONT_DANGER (1.5m)    # max -20
- 5  * severity²  if right_dist < SIDE_DANGER (0.45m)    # max -5
- 5  * severity²  if left_dist  < SIDE_DANGER (0.45m)    # max -5
-1000  se min < COLLISION_DIST (0.25m), done=True
```

### 3.3 Action space (invariato)

11 azioni discrete: angular_z ∈ [-0.8, +0.8] step 0.16. Linear_x fisso 0.5 m/s.

### 3.4 Network (invariato — `ddqn_model.py`)

DDQN: Linear(152, 300) → ReLU → Linear(300, 300) → ReLU → Linear(300, 11)

---

## 4. Componenti modificati

### 4.1 `src/my_usv/scripts/usv_env.py`

**A. Costante nuova:**

```python
DR_NOISE_STD = 0.02   # Domain randomization: gaussian noise on LIDAR (training-only)
                      # Tobin et al. 2017 — Sim-to-Real Transfer
```

**B. Spawn list M2 — D1 ricollocato:**

```python
SPAWN_LISTS = {
    1: [
        (-2.9, -2.0, 1.571),  # P1
        ( 1.0, -1.0, 1.571),  # P2
    ],
    2: [
        (-6.0,  0.0,  0.0  ),  # A1
        (-7.0,  5.0,  0.0  ),  # C2
        ( 3.5, -0.5,  1.5708), # D1 — NEW (era 1.5, 0.0, 3.142)
        (-4.5, -3.5,  0.0  ),  # F1
        (-1.5, -4.0,  1.5708), # F2
        ( 6.0,  6.0,  3.1416), # F3
    ],
    3: [
        (-2.5, -0.25, 0.0),
    ],
}
```

**C. `step_action()` con flag training + DR injection:**

```python
def step_action(self, action_index: int, training: bool = True):
    cmd = Twist()
    cmd.linear.x  = LINEAR_VEL
    cmd.angular.z = -0.8 + 0.16 * action_index
    self.vel_pub.publish(cmd)

    self._wait_sim_seconds(STEP_DT)
    rclpy.spin_once(self, timeout_sec=0.05)

    # Reward su scan PULITO (gradient coerente con reality)
    reward, done = compute_reward(self.current_scan, action_index)

    # State su scan NOISY solo in training (perception robustness)
    if training:
        noise = np.random.normal(0, DR_NOISE_STD, LIDAR_BEAMS).astype(np.float32)
        self.current_scan = np.clip(
            self.current_scan + noise, 0.0, LIDAR_MAX_RANGE
        )

    self._push_frame()
    return self.get_state(), reward, done
```

**Razionale separazione clean/noisy:**
- Reward su scan pulito = nessuna distorsione del gradiente di apprendimento
- State su scan noisy = policy impara a tollerare variabilità percettiva
- Pattern standard DR: ground-truth reward + noisy observation

### 4.2 `src/my_usv/scripts/train.py`

Chiamata a `step_action` esplicita con `training=True`:

```python
next_state, reward, done = env.step_action(action, training=True)
```

### 4.3 `src/my_usv/scripts/test.py`

Chiamata a `step_action` esplicita con `training=False`:

```python
next_state, reward, done = env.step_action(action, training=False)
```

### 4.4 `start_train_multimaze.sh`

```bash
# Era:
BLOCK_PATTERN=(2)

# Round 1 — M1 + M2 ratio 1:2:
BLOCK_PATTERN=(1 2 2)
```

Effetto: ogni 3 blocchi (300 ep), 1 blocco M1 + 2 blocchi M2 → ratio 1:2 mantenuto, M2 al 67% degli episodi.

---

## 5. Data flow

```
Training loop:
  for ep in episodes:
      maze = BLOCK_PATTERN[block_idx % len(BLOCK_PATTERN)]
      env.reset_environment(maze_id=maze, test_mode=False)
      state = env.get_state()
      while not done:
          action = agent.act(state)
          next_state, reward, done = env.step_action(action, training=True)
          # next_state = stato con LIDAR NOISY (DR attiva)
          # reward = computato su LIDAR PULITO
          buffer.push(state, action, reward, next_state, done)
          state = next_state
          agent.learn()

Test loop:
  for ep in test_episodes:
      env.reset_environment(maze_id=maze, test_mode=True)
      state = env.get_state()
      while not done:
          action = agent.act_greedy(state)
          next_state, reward, done = env.step_action(action, training=False)
          # next_state = stato con LIDAR PULITO (no DR)
          state = next_state
```

---

## 6. Test plan

### 6.1 Test unitari nuovi

**`test_usv_env.py::test_step_action_dr_training_noise_applied`**
- Setup: mock current_scan = constant value
- Call: step_action(action=5, training=True) molte volte
- Verify: get_state() ritorna LIDAR con varianza > 0 (noise iniettato)

**`test_usv_env.py::test_step_action_test_mode_no_noise`**
- Setup: mock current_scan = constant value
- Call: step_action(action=5, training=False)
- Verify: get_state() ritorna LIDAR identico a current_scan/LIDAR_MAX_RANGE (no noise)

**`test_usv_env.py::test_step_action_reward_uses_clean_scan`**
- Setup: mock scan vicino a soglia FRONT_DANGER
- Call: step_action(action=5, training=True) ripetuto
- Verify: reward identico a calcolo su scan pulito (DR non altera reward)

**`test_usv_env.py::test_d1_new_position`**
- Verify: SPAWN_LISTS[2] contiene (3.5, -0.5, ≈π/2) e NON (1.5, 0.0, ≈π)

### 6.2 Test esistenti da aggiornare

**`test_maze2_spawn_count_at_least_4`** → OK (6 ≥ 4), nessun cambio
**`test_maze2_spawn_covers_minimum_zones`** → verificare zona D ancora rappresentata da nuovo D1 (x=3.5 → zona D ✓)

### 6.3 Validation pre-training (manuale)

1. **Spawn validation runtime:** lanciare `validate_spawn.py` su nuovo D1 dopo avvio Gazebo. Atteso: OK con min > 0.40m
2. **Visual check:** `start_test_gui.sh` 1 ep su M2 con greedy policy → verificare nuovo D1 raggiungibile

### 6.4 Training run

- 5000 episodi (uguale a ddqn_enh per confronto)
- Maze pattern: M1, M2, M2 (cycling)
- MAX_STEPS = 500
- Checkpoint: salva `best_ddqn_model.pth` quando avg100 supera record
- Logging: `training_log.csv` con colonne (ep, maze_id, steps, reward, crashed, spawn, epsilon, loss)

### 6.5 Test run finale

- 90 episodi per maze (M1, M2, M3), ε=0.0
- Spawn deterministico da `TEST_SPAWN_LISTS[maze_id]` (15 ep × N spawn per maze)
- Output: `test_results.csv` con success rate per maze e per spawn

---

## 7. Success criteria

| Metrica | Baseline ddqn_enh | Target Round 1 | Stretch |
|---|---|---|---|
| M2 success rate | 51% | ≥ 45% | ≥ 55% |
| M1 success rate | 0% | ≥ 50% | ≥ 70% |
| **M3 success rate** | **0%** | **> 0%** | ≥ 15% |
| Crash rate last 100 | 60% | ≤ 60% | ≤ 50% |
| D1 NEW spawn success | n/a | ≥ 30% | ≥ 60% |

**Criteri scientifici:**
1. **Generalization confermata** se M3 > 0% in modo statisticamente significativo (su 90 ep, ≥ 5 successi = ≥ 5.5% → p<0.05 vs binomial(90, 0))
2. **No regressione M2** se M2 ≥ 45% (-6pp accettabili come trade-off per generalization)
3. **Multi-maze beneficio** se M1 ≥ 50% (recupero dopo regressione 0% da training M2-only)

**Failsafe:**
- M2 < 40% → abort, multi-maze ha disturbato troppo M2 → Round 2 = solo DR senza multi-maze
- M3 = 0% e M1 < 30% → escalation: Round 2 escalation a PPO o ricorrenza
- Crash rate last 100 > 70% → training instabile, indagare reward shaping vs multi-env interaction

---

## 8. Round 2 — decisione data-driven post Round 1

| Esito Round 1 | Round 2 |
|---|---|
| M3 ≥ 5% | Ottimizza: heading × 10 replication (STATE_DIM 152→170) + tuning |
| M3 = 0% ma M1 ≥ 50% | Escalation Track A: ricorrenza LSTM o N-step returns + Dueling |
| Tutti i target raggiunti (stretch) | Track B PPO parallelo per confronto algoritmico |
| M2 regredito sotto 40% | DR-only run senza multi-maze per isolare causa |

---

## 9. Risks & mitigations

| Rischio | Probabilità | Impatto | Mitigation |
|---|---|---|---|
| Negative transfer M1→M2 | Media | M2 cala sotto 45% | Ratio 1:2 conservativo (67% M2) |
| DR σ=0.02 troppo alto | Bassa | training instabile | Failsafe su crash rate last 100 |
| Spawn D1 NEW heading sbagliato | Bassa | D1 ancora 0% | Validation pre-training visiva |
| Multi-maze rallenta convergenza | Media | training non converge in 5000 ep | Best model checkpoint salvato (recupero migliore intermedio) |
| Test M3 sample size piccolo | Alta | varianza alta su 90 ep | Test 90 ep/maze = CI ±10pp (più che baseline 30 ep) |

---

## 10. Letteratura

- **Cobbe et al. 2019** — "Quantifying Generalization in Reinforcement Learning"
- **Tobin et al. 2017** — "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- **Peng et al. 2018** — "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
- **Mnih et al. 2015** — DQN
- **Mirowski et al. 2016** — heading necessary for indoor navigation
- **Van Hasselt et al. 2016** — DDQN target network stability
- **Feng 2021** — USV navigation reference

---

## 11. File deliverables

| File | Tipo | Stato |
|---|---|---|
| `src/my_usv/scripts/usv_env.py` | Modify | da implementare |
| `src/my_usv/scripts/train.py` | Modify (1 riga) | da implementare |
| `src/my_usv/scripts/test.py` | Modify (1 riga) | da implementare |
| `start_train_multimaze.sh` | Modify (1 riga) | da implementare |
| `src/my_usv/test/test_usv_env.py` | Modify + 4 nuovi test | da implementare |
| `analysis/maze2_geom_check.py` | Già creato | done |
| `DOCUMENTAZIONE/BRIEFING_19_05.md` | Già creato | done |
| `DOCUMENTAZIONE/ESPERIMENTI.md` | Già aggiornato | done |

---

## 12. Non-goal Round 1

Esplicitamente NON in scope:
- Heading × 10 replication (riservato Round 2)
- N-step returns
- Dueling DQN
- Noisy Networks
- PPO/SAC parallelo (Track B sospeso)
- Reward shaping changes
- BATCH_SIZE / grad_clip / loss function changes
- Recurrent networks (LSTM/GRU)
- Modifica geometria mazes

---

*Spec generato: 2026-05-19 | Branch: ddqn_round1_19_05*

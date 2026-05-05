# Training Guide — USV DDQN Collision Avoidance

**Branch:** `gym_env_claude`
**Data:** 2026-05-05

---

## Prerequisiti

| Requisito | Note |
|---|---|
| Docker Desktop (WSL2) | Deve essere avviato prima di ogni comando |
| VcXsrv / XLaunch | Richiesto per Gazebo GUI (non necessario per headless) |
| Git Bash | Shell consigliata per tutti i comandi |
| Immagine Docker `usv_rl_project` | Build iniziale descritta sotto |

### Build iniziale (una volta sola)

```bash
# Costruisci l'immagine (~2.5 GB, ROS2 Humble + PyTorch CPU + gymnasium)
docker build -t usv_rl_project .

# Compila il pacchetto ROS2 dentro il container
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash
# dentro: colcon build && exit
```

Dopo `colcon build`, le directory `build/` e `install/` esistono sull'host via volume bind. Modifiche Python si propagano istantaneamente — `colcon build` non è necessario per cambi ai file `.py`.

---

## Struttura dei branch

| Branch | Cosa contiene |
|---|---|
| `corrections_claude` | Fix reward + curriculum progressivo. Training solo via `train.py`. |
| `gym_env_claude` | Tutto di `corrections_claude` + wrapper gymnasium + `train_gym.py`. |

---

## Modalità di Training

### Modalità 1 — Curriculum DDQN (train.py)

Training completo con curriculum progressivo Phase1→Phase2. **Modalità raccomandata per la produzione.**

```bash
# Avvia o riprende il curriculum (3000 episodi, Phase1 maze1 → Phase2 30/70)
./start_training_curriculum.sh

# Ricomincia da zero (cancella checkpoint + phase.txt)
./start_training_curriculum.sh --reset
```

**Come funziona il curriculum:**

```
Phase 1 (default)
  └── Training esclusivo su Maze 1
  └── Finché avg50_reward_maze1 > 1500
  └── Quando soglia raggiunta → scrive phase.txt = "2"

Phase 2 (automatica dopo soglia)
  └── 70% Maze 2 / 30% Maze 1 per blocco
  └── Selezione casuale per blocco (non per episodio)
  └── Replay buffer Phase 1 rimane intatto → riduce forgetting
```

**Parametri chiave:**

| Parametro | Valore | File |
|---|---|---|
| Episodi totali | 3000 | `start_training_curriculum.sh` |
| Episodi per blocco | 100 | `start_training_curriculum.sh` |
| MAX_STEPS per episodio | 1000 | `train.py` |
| Soglia Phase 2 | avg50 > 1500 | `train.py` |
| Velocità Gazebo | 4x headless | `start_training_curriculum.sh` |

**Output:**

| File | Descrizione |
|---|---|
| `src/my_usv/scripts/checkpoint.pkl` | Checkpoint completo (Q-net + buffer + epsilon) |
| `src/my_usv/scripts/best_ddqn_model.pth` | Miglior modello per avg100 |
| `src/my_usv/scripts/training_log.csv` | Log episodio per episodio |
| `src/my_usv/scripts/phase.txt` | Fase corrente curriculum (`1` o `2`) |
| `logs/block_N_maze_M.log` | Log Gazebo per ogni blocco |

**Resume:** lo script riprende automaticamente dal blocco salvato in `curriculum_state.txt`. Interrompi con `Ctrl+C` — il checkpoint viene salvato ogni 20 episodi.

---

### Modalità 2 — Gymnasium DDQN (train_gym.py)

Training su singolo maze via interfaccia gymnasium. **Modalità sperimentale per validare il wrapper e testare nuovi algoritmi.**

Richiede Gazebo già avviato su un maze. Avvia prima il container con Gazebo:

```bash
# Avvia Gazebo su Maze 1 (headless, 4x)
docker run -d --rm --name usv_container \
    --volume="/$(pwd):/home/usv_ws" \
    usv_rl_project \
    bash -c "
        cd /home/usv_ws &&
        source install/setup.bash &&
        python3 src/my_usv/scripts/patch_world.py \
            /home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world 4 /tmp/world_fast.world &&
        ros2 launch my_usv spawn_robot.launch.py \
            world:=/tmp/world_fast.world x:=-3 y:=-5 yaw:=1.57 gui:=false
    "

sleep 30   # attendi che Gazebo sia pronto

# Avvia training gymnasium
docker exec usv_container bash -c "
    cd /home/usv_ws &&
    source install/setup.bash &&
    python3 src/my_usv/scripts/train_gym.py \
        --maze-id 1 \
        --episodes 3000 \
        --max-steps 1000
"
```

**Argomenti `train_gym.py`:**

| Argomento | Default | Descrizione |
|---|---|---|
| `--maze-id` | `1` | ID maze (informativo — maze dipende da Gazebo) |
| `--episodes` | `3000` | Numero episodi |
| `--max-steps` | `1000` | Limite step per episodio |
| `--continuous` | `False` | `True` → action space Box(-0.8, 0.8) |
| `--checkpoint` | `checkpoint_gym.pth` | Path checkpoint (separato da `train.py`) |

**Output:**

| File | Descrizione |
|---|---|
| `src/my_usv/scripts/checkpoint_gym.pth` | Checkpoint PyTorch state dict |
| `src/my_usv/scripts/best_gym_model.pth` | Miglior modello |

**Nota:** `checkpoint_gym.pth` e `checkpoint.pkl` sono formati diversi e non interferiscono.

---

### Modalità 3 — Gymnasium con algoritmo esterno

Per usare un algoritmo dalla repo [XinJingHao/DRL-Pytorch](https://github.com/XinJingHao/DRL-Pytorch) o qualsiasi altro algoritmo gymnasium-compatibile:

**Step 1:** Copia il file dell'algoritmo in `src/my_usv/scripts/`:
```bash
cp /path/to/xjh_ppo.py src/my_usv/scripts/
```

**Step 2:** Modifica il "swap point" in `train_gym.py`:
```python
# Sostituisci questa riga:
from train_core import DDQNAgent

# Con:
from xjh_ppo import PPO_Agent as DDQNAgent
```

**Step 3:** Assicurati che l'agente esponga questa interfaccia minima:
```python
agent.act(state)                              # → int (action index)
agent.memory.push(s, a, r, s2, terminated)   # store transition
agent.learn()                                 # → float | None (loss)
agent.step_done()                             # target net sync
agent.decay_epsilon()                         # epsilon update
```

**Step 4:** Avvia come Modalità 2.

---

## Testing

Valuta il modello salvato su tutti e 3 i maze (30 episodi ciascuno, ε=0):

```bash
./start_test.sh
```

Risultati in `src/my_usv/scripts/test_results.csv`.

**Maze 3** (`labirinto_10.world`) non è mai visto durante training — misura la generalizzazione.

---

## Test unitari

Non richiedono Gazebo. Verificano logica pura (LIDAR, reward, DDQN, gymnasium wrapper):

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v"
```

**Suite:** 52 test, tutti GREEN.

| File test | Cosa copre |
|---|---|
| `test_usv_logic.py` | LIDAR processing, reward function (space bonus, danger zones) |
| `test_ddqn_model.py` | Forward pass, shape, gradienti |
| `test_replay_buffer.py` | Push, capacity FIFO, sample shape |
| `test_agent.py` | Epsilon, greedy action, learn, target sync |
| `test_usv_gym_env.py` | Spaces, reset, step, terminated/truncated, action mapping |

---

## Troubleshooting

| Problema | Causa | Fix |
|---|---|---|
| `docker: Error: No such container` | Container non avviato | Attendi Gazebo (30s) |
| LIDAR sync failures | Gazebo speed > 4x | Mantieni `GAZEBO_SPEED=4` |
| Robot gira in cerchio | Reward hacking (vecchio training) | `--reset` e riparti |
| Phase 2 non scatta | `maze1_window` si azzera dopo crash | Normale — serve finestra di 50 ep consecutivi su maze 1 |
| Container bloccato | Training interrotto male | `docker rm -f usv_container` |

---

## Reward function (post-fix)

```
reward = +5.0                                           # sopravvivenza
       + 2.0 * mean(scan) / 5.0                        # space bonus (max +2.0)
       - |action - 5| * 0.02                           # steering (anti-oscillazione)
       - 20.0 * severity²  (se fronte < 3.0m)         # front danger
       - 5.0  * severity²  (se lato < 0.45m)          # side danger
       = -1000.0 (terminale)                            # collisione
```

| Costante | Valore | Motivazione |
|---|---|---|
| `FRONT_DANGER` | 3.0m | Robot vede muro 15 step prima (era 1.5m) |
| `SPACE_BONUS_WEIGHT` | 2.0 | Incentiva spazio aperto, risolve spinning |
| Steering penalty | 0.02 | Era 0.1 — meno vincolo comportamentale |
| Front exponent | quadratico | Era cubico — segnale 4.5x più forte a 2.0m |

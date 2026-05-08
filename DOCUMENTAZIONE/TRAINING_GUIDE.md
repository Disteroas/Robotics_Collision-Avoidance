# Training Guide — USV DDQN Collision Avoidance

**Branch:** `corrections_claude`
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
# Costruisci l'immagine (~2.5 GB, ROS2 Humble + PyTorch CPU)
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
| `corrections_claude` | Fix reward + curriculum progressivo. Training via `train.py`. **(questo branch)** |
| `gym_env_claude` | Tutto di questo branch + wrapper gymnasium + `train_gym.py`. |

---

## Training — Curriculum DDQN

Unica modalità di training su questo branch. Curriculum progressivo Phase1→Phase2 su 3000 episodi.

### Avvio

```bash
# Avvia o riprende il curriculum
./start_training_curriculum.sh

# Ricomincia da zero (cancella checkpoint + phase.txt + stato curriculum)
./start_training_curriculum.sh --reset
```

### Come funziona il curriculum

```
Phase 1 (default, parte da subito)
  └── Training esclusivo su Maze 1
  └── Finché avg50_reward_maze1 > 1500
  └── Quando soglia raggiunta → scrive phase.txt = "2"
  └── Atteso intorno a ep 1200–1400 (basato su dati storici)

Phase 2 (automatica dopo soglia)
  └── 70% Maze 2 / 30% Maze 1 per blocco (selezione casuale)
  └── Replay buffer Phase 1 rimane intatto → riduce catastrophic forgetting
  └── Switch per blocco (non per episodio) — richiede restart Gazebo
```

**Perché progressivo invece di alternanza fissa:**
Il Maze 1 è molto più semplice del Maze 2. L'alternanza a blocchi fissi (precedente approccio) causava oscillazione avg100 di ±250 ad ogni switch — firma classica di catastrophic forgetting. Il curriculum progressivo consolida il Maze 1 prima di introdurre il Maze 2.

### Parametri chiave

| Parametro | Valore | File |
|---|---|---|
| Episodi totali | 3000 | `start_training_curriculum.sh` |
| Episodi per blocco | 100 | `start_training_curriculum.sh` |
| MAX_STEPS per episodio | 1000 | `train.py` |
| Soglia Phase 2 | avg50 > 1500 | `train.py` |
| Velocità Gazebo | 4x headless | `start_training_curriculum.sh` |
| Soglia finestra | 50 episodi | `train.py` (`PHASE1_WINDOW`) |

### Output

| File | Descrizione |
|---|---|
| `src/my_usv/scripts/checkpoint.pkl` | Checkpoint completo (Q-net + target-net + optimizer + replay buffer + epsilon) |
| `src/my_usv/scripts/best_ddqn_model.pth` | Miglior modello per avg100 |
| `src/my_usv/scripts/training_log.csv` | Log episodio per episodio (ep, maze, reward, avg100, epsilon, loss, crash) |
| `src/my_usv/scripts/phase.txt` | Fase corrente curriculum (`1` o `2`) |
| `logs/block_N_maze_M.log` | Log Gazebo per ogni blocco |

### Resume e interruzione

Lo script riprende automaticamente dal blocco salvato in `curriculum_state.txt`. Interrompi con `Ctrl+C` — il checkpoint viene salvato ogni 20 episodi (perdita massima: 20 ep).

```bash
# Controlla la fase attuale
cat src/my_usv/scripts/phase.txt

# Controlla il blocco corrente
cat src/my_usv/scripts/curriculum_state.txt

# Forza transizione a Phase 2 manualmente
echo "2" > src/my_usv/scripts/phase.txt
```

### Avvio manuale (debug)

Per eseguire un singolo blocco senza lo script bash:

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

sleep 30

# Esegui blocco 0-100 su Maze 1
docker exec usv_container bash -c "
    cd /home/usv_ws &&
    source install/setup.bash &&
    python3 src/my_usv/scripts/train.py \
        --start-ep 0 \
        --end-ep 100 \
        --maze-id 1 \
        --checkpoint src/my_usv/scripts/checkpoint.pkl \
        --phase-file src/my_usv/scripts/phase.txt
"

# Ferma container
docker rm -f usv_container
```

---

## Testing

Valuta il modello salvato su tutti e 3 i maze (30 episodi ciascuno, ε=0):

```bash
./start_test.sh
```

Risultati in `src/my_usv/scripts/test_results.csv`.

| Maze | Tipo | Spawn |
|---|---|---|
| Maze 1 (`labirinto_9a.world`) | Training | x=-3, y=-5, yaw=1.57 |
| Maze 2 (`labirinto_9b.world`) | Training | x=-6, y=0, yaw=0 |
| Maze 3 (`labirinto_10.world`) | **Test (mai visto)** | x=-2, y=-1, yaw=0 |

**Maze 3 non è mai visto durante training** — misura la generalizzazione del modello.

---

## Test unitari

Non richiedono Gazebo. Verificano logica pura (LIDAR processing, reward, DDQN):

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v"
```

**Suite:** 41 test, tutti GREEN.

| File test | Cosa copre |
|---|---|
| `test_usv_logic.py` | LIDAR min-pooling, NaN handling, reward (space bonus, front/side danger) |
| `test_ddqn_model.py` | Forward pass, output shape, gradienti |
| `test_replay_buffer.py` | Push, capacity FIFO, sample shape |
| `test_agent.py` | Epsilon decay, greedy action, learn, target network sync |

---

## Architettura reward function

```
reward = +5.0                                           # sopravvivenza base
       + 2.0 * mean(scan) / 5.0                        # space bonus (max +2.0)
       - |action - 5| * 0.02                           # steering (anti-oscillazione)
       - 20.0 * severity²  (se fronte < 3.0m)         # front danger quadratico
       - 5.0  * severity²  (se lato < 0.45m)          # side danger quadratico
       = -1000.0 (terminale)                            # collisione
```

| Costante | Valore | Motivazione |
|---|---|---|
| `FRONT_DANGER` | 3.0m | Era 1.5m — robot vede il muro 15 step prima |
| `SPACE_BONUS_WEIGHT` | 2.0 | Incentiva spazio aperto, risolve reward hacking (spinning) |
| Steering penalty | 0.02 | Era 0.1 — ridotto per non vincolare eccessivamente la svolta |
| Front exponent | quadratico | Era cubico — segnale 4.5× più forte a 2.0m di distanza |
| `MAX_STEPS` | 1000 | Era 500 — evita falsi positivi "robot sopravvive immobile" |

---

## Troubleshooting

| Problema | Causa | Fix |
|---|---|---|
| `docker: No such container` | Container non avviato | Attendi Gazebo (30s) |
| LIDAR sync failures | `GAZEBO_SPEED` > 4 | Mantieni `GAZEBO_SPEED=4` |
| Robot gira in cerchio | Reward hacking (modello vecchio) | `--reset` e riparti |
| Phase 2 non scatta | `maze1_window` si azzera dopo crash o resume | Normale — servono 50 ep consecutivi su maze 1 sopra soglia |
| Container bloccato | Training interrotto male | `docker rm -f usv_container` |
| `colcon build` richiesto | Modifiche a CMakeLists.txt o package.xml | Ri-esegui build nel container |

# Multi-Maze Interleaved Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementare training DDQN interleaved M1+M2 (ratio 1:2) con random spawn in entrambi i maze, 5000 episodi, su branch `merge11_05`.

**Architecture:** Gazebo riavvia ogni 200 episodi per caricare il world file del maze corrente. Un nuovo shell script orchestra 25 blocchi seguendo il pattern (M1, M2, M2) ciclico. Il checkpoint è condiviso tra tutti i blocchi — epsilon e replay buffer continuano senza reset.

**Tech Stack:** Python 3, PyTorch CPU, ROS2 Humble, Gazebo, Docker, Bash.

---

## File Structure

| File | Ruolo modifica |
|------|----------------|
| `src/my_usv/scripts/usv_env.py` | `SPAWN_LISTS[1]`: 8 → 16 punti (zone A-F) |
| `src/my_usv/scripts/train.py` | Rimuove phase transition; aggiunge `--total-ep` CLI arg |
| `start_train_multimaze.sh` | Nuovo script: 25 blocchi × 200 ep, pattern M1/M2/M2 |

**Invariati:** `train_core.py`, `ddqn_model.py`, `usv_logic.py`, `test.py`, `start_test.sh`

---

## Task 1: Crea branch merge11_05

**Files:** nessuno

- [ ] **Step 1: Crea e checkout branch**

```bash
git checkout -b merge11_05
```

Expected: `Switched to a new branch 'merge11_05'`

- [ ] **Step 2: Verifica branch attivo**

```bash
git branch --show-current
```

Expected: `merge11_05`

---

## Task 2: Espandi SPAWN_LISTS[1] in usv_env.py

**Files:**
- Modify: `src/my_usv/scripts/usv_env.py:15-55`

- [ ] **Step 1: Verifica stato attuale (8 punti)**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src/my_usv/scripts')
# import senza rclpy
import importlib, types
rclpy_mock = types.ModuleType('rclpy'); sys.modules['rclpy'] = rclpy_mock
for sub in ['rclpy.node','rclpy.parameter','rclpy.time','rclpy.Parameter']:
    sys.modules[sub] = types.ModuleType(sub)
# stub dipendenze ROS
for mod in ['geometry_msgs','geometry_msgs.msg','sensor_msgs','sensor_msgs.msg',
            'std_srvs','std_srvs.srv','gazebo_msgs','gazebo_msgs.srv']:
    sys.modules[mod] = types.ModuleType(mod)
    if '.msg' in mod or '.srv' in mod:
        parent = sys.modules[mod.rsplit('.',1)[0]]
        setattr(parent, mod.split('.')[-1], sys.modules[mod])
import usv_env
print(len(usv_env.SPAWN_LISTS[1]))
"
```

Expected: `8`

- [ ] **Step 2: Sostituisci SPAWN_LISTS[1] con 16 punti**

In `src/my_usv/scripts/usv_env.py`, sostituisci le righe 15-55 (dalla riga `SPAWN_LISTS = {` fino alla riga `}` che chiude il dict) con:

```python
# Random spawn positions per maze — validate with ./test_spawns.sh before training
SPAWN_LISTS = {
    1: [
        # Zone A: sud (4 spawn) — original 4 points kept
        (-3.0, -5.0,  1.57 ),  # A1: south open, heading N
        ( 0.0, -4.5,  1.57 ),  # A2: centre-south, heading N
        ( 2.5, -5.0,  1.57 ),  # A3: right-south, heading N
        (-1.5, -5.0,  0.0  ),  # A4: south open, heading E

        # Zone B: sinistra (3 spawn)
        (-2.9, -2.0,  1.57 ),  # B1: left channel entry, heading N
        (-2.9,  0.5,  0.0  ),  # B2: left channel mid, heading E
        (-2.5, -3.5,  0.785),  # B3: left-mid, heading NE

        # Zone C: centro (3 spawn)
        ( 0.5, -2.5,  1.57 ),  # C1: centre-bottom, heading N
        ( 0.0, -1.0,  3.142),  # C2: centre, heading W
        ( 0.0,  0.0,  1.57 ),  # C3: centre, heading N

        # Zone D: destra (2 spawn)
        ( 2.5, -2.0,  1.57 ),  # D1: right outer, heading N
        ( 2.5,  0.0,  3.142),  # D2: right, heading W

        # Zone E: superiore (2 spawn)
        (-1.0,  1.5,  0.0  ),  # E1: upper-left, heading E
        ( 1.0,  1.5,  3.142),  # E2: upper-right, heading W

        # Zone F: copertura extra (2 spawn)
        (-3.0, -4.0,  0.0  ),  # F1: south-left, heading E
        ( 1.5, -4.0,  1.57 ),  # F2: south-right, heading N
    ],
    2: [
        # Zone A: ingresso sinistro (2 spawn) — validated min>=0.43m
        (-6.0,  0.0,  0.0  ),  # A1: heading E  — min=1.352m
        (-6.5, -0.5,  0.0  ),  # A2: heading E  — min=1.803m

        # Zone B: centro-sinistra (3 spawn) — validated min>=0.43m
        (-4.5,  0.5,  0.0  ),  # B1: heading E  — min=0.995m
        (-4.0, -1.0,  1.571),  # B2: heading N  — min=0.523m
        (-4.5,  1.5,  2.356),  # B3: heading NW — min=0.497m

        # Zone C: centro (3 spawn) — validated min>=0.43m
        (-2.5,  1.0,  0.0  ),  # C1: heading E  — min=0.434m
        (-7.0,  5.0,  0.0  ),  # C2: heading E  — min=0.860m
        (-2.0, -1.0,  0.785),  # C3: heading NE — min=0.795m

        # Zone D: centro-destra (3 spawn) — validated min>=0.43m
        ( 1.5,  0.0,  3.142),  # D1: heading W  — min=0.693m
        ( 0.5, -2.0,  1.571),  # D2: heading N  — min=0.430m
        ( 3.5,  0.5,  4.712),  # D3: heading S  — min=0.780m

        # Zone E: superiore (2 spawn) — validated min>=0.43m
        (-3.0,  3.0,  0.0  ),  # E1: heading E  — min=0.890m
        ( 0.0,  3.5,  3.142),  # E2: heading W  — min=0.650m

        # Zone F: inferiore (3 spawn) — validated min>=0.43m
        (-4.5, -3.5,  0.0  ),  # F1: heading E  — min=1.162m
        (-1.5, -4.0,  1.571),  # F2: heading N  — min=1.008m
        ( 6.0,  6.0,  3.142),  # F3: heading W  — min=0.456m
    ],
}
```

- [ ] **Step 3: Verifica 16 punti e tipi corretti**

```bash
python3 -c "
import sys, types
rclpy_mock = types.ModuleType('rclpy'); sys.modules['rclpy'] = rclpy_mock
for sub in ['rclpy.node','rclpy.parameter','rclpy.time']:
    sys.modules[sub] = types.ModuleType(sub)
for mod in ['geometry_msgs','geometry_msgs.msg','sensor_msgs','sensor_msgs.msg',
            'std_srvs','std_srvs.srv','gazebo_msgs','gazebo_msgs.srv']:
    sys.modules[mod] = types.ModuleType(mod)
    if '.' in mod:
        setattr(sys.modules[mod.rsplit('.',1)[0]], mod.split('.')[-1], sys.modules[mod])
sys.path.insert(0, 'src/my_usv/scripts')
import usv_env
assert len(usv_env.SPAWN_LISTS[1]) == 16, f'M1: expected 16, got {len(usv_env.SPAWN_LISTS[1])}'
assert len(usv_env.SPAWN_LISTS[2]) == 16, f'M2: expected 16, got {len(usv_env.SPAWN_LISTS[2])}'
for i, pt in enumerate(usv_env.SPAWN_LISTS[1]):
    assert len(pt) == 3, f'M1 point {i}: expected 3-tuple'
    assert all(isinstance(v, float) for v in pt), f'M1 point {i}: all values must be float'
print('OK: SPAWN_LISTS[1]=16 pts, SPAWN_LISTS[2]=16 pts, types correct')
"
```

Expected: `OK: SPAWN_LISTS[1]=16 pts, SPAWN_LISTS[2]=16 pts, types correct`

- [ ] **Step 4: Commit**

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "feat: expand M1 spawn points from 8 to 16 (zones A-F)"
```

---

## Task 3: Modifica train.py — rimuovi phase transition, aggiungi --total-ep

**Files:**
- Modify: `src/my_usv/scripts/train.py`

Riferimento: il file attuale ha queste sezioni da rimuovere/modificare:
- Righe 41-43: costanti `PHASE2_THRESHOLD`, `PHASE1_WINDOW`, `EPSILON_RESET_P2`
- Righe 53-55: argomento `--phase-file` in `parse_args`
- Righe 58-60: funzione `_write_phase`
- Riga 74: `maze1_window = deque(maxlen=PHASE1_WINDOW)`
- Riga 84: `total_ep = 3000` hardcoded
- Righe 139-148: blocco `if args.maze_id == 1:` (phase detection)

- [ ] **Step 1: Rimuovi le 3 costanti phase transition (righe 41-43)**

Trova e rimuovi:
```python
PHASE2_THRESHOLD  = 1500   # avg reward maze1 su finestra 50 ep per passare a Phase 2
PHASE1_WINDOW     = 50     # dimensione finestra per calcolo threshold
EPSILON_RESET_P2  = 0.5    # floor ε quando Phase 2 si attiva (Narvekar et al. 2020)
```

- [ ] **Step 2: Rimuovi argomento --phase-file da parse_args e aggiungi --total-ep**

Trova:
```python
    p.add_argument('--phase-file', type=str,
                   default='src/my_usv/scripts/phase.txt')
    return p.parse_args()
```

Sostituisci con:
```python
    p.add_argument('--total-ep',   type=int, default=5000)
    return p.parse_args()
```

- [ ] **Step 3: Rimuovi funzione _write_phase**

Trova e rimuovi:
```python
def _write_phase(phase_file: str, phase: int) -> None:
    with open(phase_file, 'w') as f:
        f.write(str(phase))
```

- [ ] **Step 4: Rimuovi maze1_window dalla inizializzazione**

Trova e rimuovi:
```python
    maze1_window = deque(maxlen=PHASE1_WINDOW)
```

- [ ] **Step 5: Sostituisci total_ep hardcoded con args.total_ep**

Trova:
```python
    best_avg = -float('inf')
    total_ep = 3000
```

Sostituisci con:
```python
    best_avg = -float('inf')
    total_ep = args.total_ep
```

- [ ] **Step 6: Rimuovi blocco phase detection**

Trova e rimuovi l'intero blocco (dopo `agent.decay_epsilon()` e prima di `avg100 = ...`):
```python
        # Phase detection: monitora maze 1 per transizione Phase 1 → Phase 2
        if args.maze_id == 1:
            maze1_window.append(ep_rew)
            if (len(maze1_window) == PHASE1_WINDOW
                    and float(np.mean(maze1_window)) > PHASE2_THRESHOLD):
                phase_path = os.path.abspath(args.phase_file)
                if not os.path.exists(phase_path) or open(phase_path).read().strip() == '1':
                    _write_phase(phase_path, 2)
                    agent.epsilon = max(agent.epsilon, EPSILON_RESET_P2)
                    print(f"  PHASE 2 sbloccata! avg50_maze1={float(np.mean(maze1_window)):.1f} > {PHASE2_THRESHOLD}")
                    print(f"  ε reset → {agent.epsilon:.3f}")
```

- [ ] **Step 7: Verifica sintassi e assenza phase transition**

```bash
python3 -c "
import ast, sys
with open('src/my_usv/scripts/train.py') as f:
    src = f.read()
# Syntax check
try:
    ast.parse(src)
    print('Syntax OK')
except SyntaxError as e:
    print(f'SYNTAX ERROR: {e}'); sys.exit(1)
# Phase transition removed
for token in ['PHASE2_THRESHOLD', 'PHASE1_WINDOW', 'EPSILON_RESET_P2',
              'maze1_window', '_write_phase', 'phase_file']:
    assert token not in src, f'Token ancora presente: {token}'
# --total-ep present
assert 'total-ep' in src, '--total-ep argument missing'
assert 'args.total_ep' in src, 'args.total_ep missing'
print('Phase transition removed OK, --total-ep present OK')
"
```

Expected:
```
Syntax OK
Phase transition removed OK, --total-ep present OK
```

- [ ] **Step 8: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "refactor: remove phase transition logic from train.py, add --total-ep arg"
```

---

## Task 4: Crea start_train_multimaze.sh

**Files:**
- Create: `start_train_multimaze.sh`

- [ ] **Step 1: Crea il file**

Crea `start_train_multimaze.sh` nella root del progetto con il seguente contenuto:

```bash
#!/bin/bash
# =============================================================================
#  start_train_multimaze.sh  —  Multi-maze interleaved training (M1:M2 = 1:2)
#
#  5000 episodi, 25 blocchi × 200 ep, pattern (M1, M2, M2) ciclico.
#  Gazebo riavvia ogni blocco per caricare il world file del maze corrente.
#  Checkpoint condiviso: epsilon e replay buffer continuano senza reset.
#
#  Uso:
#    ./start_train_multimaze.sh           # riprende da checkpoint esistente
#    ./start_train_multimaze.sh --reset   # cancella tutto e riparte da zero
#
#  PREREQUISITO: colcon build eseguito almeno una volta.
# =============================================================================

GAZEBO_SPEED=5
GAZEBO_WAIT=30
TOTAL_BLOCKS=25
BLOCK_SIZE=200
BLOCK_PATTERN=(1 2 2)

WORLD_PATH_1="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH_2="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
SPAWN_ARGS_1="x:=-3 y:=-5 yaw:=1.57"
SPAWN_ARGS_2="x:=-6 y:=0 yaw:=0"

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="${SCRIPTS_CTR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))

mkdir -p "$(pwd)/logs"

if [[ "$1" == "--reset" ]]; then
    echo "  --reset: rimozione checkpoint e log precedenti..."
    rm -f src/my_usv/scripts/checkpoint.pkl
    rm -f src/my_usv/scripts/training_log.csv
    rm -f src/my_usv/scripts/best_ddqn_model.pth
    rm -f src/my_usv/scripts/phase.txt
    echo "  Reset completato."
fi

echo ""
echo "============================================================"
echo "  USV DDQN — MULTI-MAZE INTERLEAVED TRAINING"
echo "============================================================"
echo "  Maze pattern : M1/M2/M2 (ratio 1:2)"
echo "  Episodi tot  : ${TOTAL_EP}"
echo "  Blocchi      : ${TOTAL_BLOCKS} x ${BLOCK_SIZE} ep"
echo "  Gazebo speed : ${GAZEBO_SPEED}x headless"
echo "  Checkpoint   : src/my_usv/scripts/checkpoint.pkl"
echo "============================================================"
echo ""

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    pattern_idx=$(( (b - 1) % ${#BLOCK_PATTERN[@]} ))
    MAZE_ID=${BLOCK_PATTERN[$pattern_idx]}

    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    if [[ $MAZE_ID -eq 1 ]]; then
        WORLD_PATH="$WORLD_PATH_1"
        SPAWN_ARGS="$SPAWN_ARGS_1"
    else
        WORLD_PATH="$WORLD_PATH_2"
        SPAWN_ARGS="$SPAWN_ARGS_2"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Maze ${MAZE_ID} | ep $((START_EP+1))-${END_EP}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container 2>/dev/null
    sleep 1

    LOG_FILE="$(pwd)/logs/multimaze_block_$(printf '%02d' $b)_maze_${MAZE_ID}.log"

    docker run -d --rm --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/patch_world.py \
                '${WORLD_PATH}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
            ros2 launch my_usv spawn_robot.launch.py \
                world:=${PATCHED_WORLD} \
                ${SPAWN_ARGS} \
                gui:=false
        " > "$LOG_FILE" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "  ERRORE: avvio container fallito al blocco ${b}."
        exit 1
    fi

    echo "  Attendo ${GAZEBO_WAIT}s avvio Gazebo..."
    sleep "$GAZEBO_WAIT"

    running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
    if [[ "$running" != "true" ]]; then
        echo "  ERRORE: Gazebo crashato al blocco ${b}. Ultimi log:"
        tail -20 "$LOG_FILE"
        exit 1
    fi

    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/train.py \
                --maze-id    ${MAZE_ID} \
                --start-ep   ${START_EP} \
                --end-ep     ${END_EP} \
                --total-ep   ${TOTAL_EP} \
                --checkpoint ${CHECKPOINT_CTR}
        "

    EXIT_CODE=$?
    docker rm -f usv_container 2>/dev/null

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  ERRORE: train.py exit ${EXIT_CODE} al blocco ${b}. Interruzione."
        exit 1
    fi

    echo "  Blocco ${b}/${TOTAL_BLOCKS} completato."
done

echo ""
echo "============================================================"
echo "  TRAINING COMPLETATO — ${TOTAL_EP} episodi"
echo "============================================================"
echo "  Modello: src/my_usv/scripts/best_ddqn_model.pth"
echo "  Log:     src/my_usv/scripts/training_log.csv"
echo "  Blocchi: logs/multimaze_block_*.log"
echo "============================================================"
```

- [ ] **Step 2: Rendi eseguibile**

```bash
chmod +x start_train_multimaze.sh
```

- [ ] **Step 3: Verifica sintassi bash**

```bash
bash -n start_train_multimaze.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 4: Verifica il pattern blocchi (dry-run)**

```bash
bash -c "
BLOCK_PATTERN=(1 2 2)
TOTAL_BLOCKS=25
BLOCK_SIZE=200
m1=0; m2=0
for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    idx=\$(( (b-1) % \${#BLOCK_PATTERN[@]} ))
    maze=\${BLOCK_PATTERN[\$idx]}
    start=\$(( (b-1)*BLOCK_SIZE + 1 ))
    end=\$(( b*BLOCK_SIZE ))
    echo \"  Block \$b: Maze \$maze  ep \$start-\$end\"
    [[ \$maze -eq 1 ]] && (( m1++ )) || (( m2++ ))
done
echo \"\"
echo \"  M1 blocks: \$m1 ($(( m1*200 )) ep)\"
echo \"  M2 blocks: \$m2 ($(( m2*200 )) ep)\"
echo \"  Ratio M1:M2 = \$m1:\$m2\"
"
```

Expected output (ultimi 4 righe):
```
  M1 blocks: 9 (1800 ep)
  M2 blocks: 16 (3200 ep)
  Ratio M1:M2 = 9:16
```

- [ ] **Step 5: Commit**

```bash
git add start_train_multimaze.sh
git commit -m "feat: add start_train_multimaze.sh — interleaved M1/M2 training, 25 blocks x 200 ep"
```

---

## Task 5: Validazione spawn M1 (richiede Gazebo running)

**Files:** nessuno (validazione operativa)

⚠️ Questo task richiede Gazebo attivo. Non può essere eseguito offline.

- [ ] **Step 1: Avvia Gazebo su Maze 1**

```bash
docker rm -f usv_container 2>/dev/null
docker run -d --rm --name usv_container \
    --volume="/$(pwd):/home/usv_ws" \
    usv_rl_project \
    bash -c "
        cd /home/usv_ws && source install/setup.bash && \
        python3 src/my_usv/scripts/patch_world.py \
            /home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world \
            5 /tmp/world_fast.world && \
        ros2 launch my_usv spawn_robot.launch.py \
            world:=/tmp/world_fast.world x:=-3 y:=-5 yaw:=1.57 gui:=false
    "
sleep 30
```

- [ ] **Step 2: Esegui test spawn M1**

```bash
./test_spawns.sh 1
```

Expected: tutti e 16 i punti mostrano `min_lidar >= 0.40m`. Punti con `UNSAFE` vanno sostituiti con coordinate alternative nello stesso intorno (±0.3m) e ritestati.

- [ ] **Step 3: Se necessario, correggi punti unsafe in usv_env.py e ri-committa**

Se uno o più punti risultano unsafe (min_lidar < 0.40m), modificali in `src/my_usv/scripts/usv_env.py` spostandoli di ±0.3m in x o y e ri-esegui Step 2. Poi:

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "fix: adjust unsafe M1 spawn points after validation"
```

- [ ] **Step 4: Kill container**

```bash
docker rm -f usv_container 2>/dev/null
```

---

## Task 6: Avvio training

**Files:** nessuno (esecuzione operativa)

- [ ] **Step 1: Avvia training da zero**

```bash
./start_train_multimaze.sh --reset
```

- [ ] **Step 2: Monitora blocco 1 (M1)**

Nei primi 200 episodi verifica che il log mostri:
- `[M1]` nel nome blocco
- `ε` che scende da ~1.0 verso ~0.82 (non deve essere già 0.05)
- Crash rate alta inizialmente (normale)

- [ ] **Step 3: Verifica transizione blocco 1→2 (M1→M2)**

Quando il blocco 1 termina, verifica nel log:
```
  Blocco 1/25 completato.
  Blocco 2/25 | Maze 2 | ep 201-400
```
E che Gazebo riparta con il world file di Maze 2.

- [ ] **Step 4: Verifica checkpoint condiviso dopo blocco 2**

```bash
python3 -c "
import pickle
with open('src/my_usv/scripts/checkpoint.pkl', 'rb') as f:
    d = pickle.load(f)
print(f'ep: {d[\"episode\"]}')
print(f'epsilon: {d[\"epsilon\"]:.4f}')
print(f'buffer size: {len(d[\"replay_buffer\"])}')
print(f'crashes: {d[\"crashes\"]}')
"
```

Expected dopo blocco 2 (ep 400):
```
ep: 400
epsilon: ~0.670  (0.999^400 ≈ 0.670)
buffer size: >400  (potenzialmente migliaia se episodi lunghi)
crashes: <400
```

---

## Self-Review

### Spec coverage check

| Requisito spec | Task che lo implementa |
|---------------|----------------------|
| SPAWN_LISTS[1] 8→16 punti | Task 2 |
| Rimuovi phase transition logic | Task 3 steps 1,4,6 |
| total_ep da CLI (default 5000) | Task 3 steps 2,5 |
| Script 25 blocchi × 200 ep | Task 4 step 1 |
| Pattern M1/M2/M2 ciclico | Task 4 step 1 (BLOCK_PATTERN) |
| --reset flag | Task 4 step 1 |
| Checkpoint condiviso | Task 4 step 1 (CHECKPOINT_CTR costante) |
| Validazione spawn M1 | Task 5 |
| Branch merge11_05 | Task 1 |

### Note implementative

- `SPAWN_ARGS_1` e `SPAWN_ARGS_2` nello script sono i parametri di spawn iniziale per Gazebo (dove il robot appare al caricamento del world). Sono sovrascritt immediatamente da `reset_environment()` in Python — il valore esatto non è critico ma deve essere una posizione valida nel maze.
- Il log di ogni blocco va in `logs/multimaze_block_NN_maze_M.log` (zero-padded per ordinamento corretto: `01`, `02`, ..., `25`).
- Il training può essere interrotto (Ctrl+C) e ripreso: il checkpoint salva ogni 20 episodi. Alla ripresa, lo script parte dal blocco che ha il checkpoint più recente — ma poiché train.py verifica `if last_ep >= args.end_ep`, i blocchi già completati vengono saltati automaticamente.

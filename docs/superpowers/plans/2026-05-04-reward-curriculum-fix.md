# Reward Function & Curriculum Learning Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix reward hacking e catastrophic forgetting sostituendo il survival-only reward con un open-space bonus, estendendo la zona pericolo frontale, e implementando curriculum progressivo Phase1→Phase2.

**Architecture:** La logica pura di reward vive in `usv_logic.py` (testabile senza ROS2). Il rilevamento della threshold Phase1→Phase2 vive in `train.py` e scrive `phase.txt`. Lo script bash legge `phase.txt` per scegliere il maze ad ogni blocco.

**Tech Stack:** Python 3.10, NumPy, pytest (dentro Docker), Bash

---

## File Map

| File | Tipo | Cosa cambia |
|---|---|---|
| `src/my_usv/scripts/usv_logic.py` | Modifica | `FRONT_DANGER` 1.5→3.0, aggiungi `SPACE_BONUS_WEIGHT=2.0`, `compute_reward` quadratico + space bonus, steering 0.1→0.02 |
| `src/my_usv/test/test_usv_logic.py` | Modifica | Aggiorna 5 test esistenti, aggiungi 3 nuovi |
| `src/my_usv/scripts/train.py` | Modifica | `MAX_STEPS` 500→1000, aggiungi `PHASE2_THRESHOLD`, `--phase-file` arg, scrittura `phase.txt` |
| `start_training_curriculum.sh` | Modifica | Rimuovi `MAZE_SEQUENCE`, aggiungi lettura `phase.txt`, Phase1=solo maze1, Phase2=random 30/70 |

---

## Task 1: Aggiorna reward function (TDD)

**Files:**
- Modify: `src/my_usv/scripts/usv_logic.py`
- Modify: `src/my_usv/test/test_usv_logic.py`

### Step 1 — Aggiorna i test esistenti che cambieranno valore

I seguenti test passeranno a valori diversi dopo il fix. Aggiornali PRIMA di toccare `usv_logic.py` così diventano RED.

Apri `src/my_usv/test/test_usv_logic.py` e sostituisci i test indicati:

```python
# SOSTITUISCI test_clear_path_straight_returns_base_reward
def test_clear_path_straight_returns_base_reward():
    # action 5, tutto libero: +5.0 base + 2.0 space bonus (mean=5.0/5.0*2.0) = 7.0
    reward, done = compute_reward(_clear_scan(), action_index=5)
    assert reward == pytest.approx(7.0)
    assert done is False


# SOSTITUISCI test_hard_left_has_steering_penalty
def test_hard_left_has_steering_penalty():
    # action 0: |0-5| * 0.02 = 0.1 penalità. Spazio libero: space_bonus=2.0
    reward, _ = compute_reward(_clear_scan(), action_index=0)
    assert reward == pytest.approx(6.9)


# SOSTITUISCI test_hard_right_has_same_penalty_as_hard_left
def test_hard_right_has_same_penalty_as_hard_left():
    r_left, _  = compute_reward(_clear_scan(), action_index=0)
    r_right, _ = compute_reward(_clear_scan(), action_index=10)
    assert r_left == pytest.approx(r_right)


# SOSTITUISCI test_front_danger_severity_is_cubic (rinomina a quadratic)
def test_front_danger_severity_is_quadratic():
    # FRONT_DANGER=3.0, midpoint=(3.0+0.25)/2=1.625 → severity=0.5
    # penalty = 20*(0.5**2) = 5.0
    midpoint = (FRONT_DANGER + COLLISION_DIST) / 2   # 1.625 m
    scan = _clear_scan()
    scan[15:35] = midpoint
    # mean(scan) = (30*5.0 + 20*1.625) / 50 = 3.65
    expected_space_bonus = 2.0 * ((30 * LIDAR_MAX_RANGE + 20 * midpoint) / LIDAR_BEAMS) / LIDAR_MAX_RANGE
    expected = 5.0 + expected_space_bonus - 5.0  # base + bonus - front_penalty
    reward, _ = compute_reward(scan, action_index=5)
    assert reward == pytest.approx(expected, abs=1e-4)


# SOSTITUISCI test_front_danger_reduces_reward
def test_front_danger_reduces_reward():
    scan = _clear_scan()
    scan[15:35] = 2.0  # fronte < FRONT_DANGER(3.0m)
    reward_danger, done = compute_reward(scan, action_index=5)
    reward_clear, _     = compute_reward(_clear_scan(), action_index=5)
    assert reward_danger < reward_clear
    assert not done
```

- [ ] **Step 1a: Sostituisci i 5 test in `test_usv_logic.py`** con il codice sopra (rimuovi `test_front_danger_severity_is_cubic`, aggiungi `test_front_danger_severity_is_quadratic`).

- [ ] **Step 1b: Aggiungi 3 nuovi test in fondo al file**

```python
def test_space_bonus_increases_with_open_space():
    scan_clear = _clear_scan()                          # mean=5.0 → bonus=2.0
    scan_tight = np.ones(LIDAR_BEAMS) * 0.5            # mean=0.5 → bonus=0.2
    r_clear, _ = compute_reward(scan_clear, action_index=5)
    r_tight, _ = compute_reward(scan_tight, action_index=5)
    assert r_clear > r_tight


def test_space_bonus_max_in_fully_clear_scan():
    # mean=5.0, bonus = 2.0 * 5.0/5.0 = 2.0
    reward, _ = compute_reward(_clear_scan(), action_index=5)
    assert reward == pytest.approx(7.0)


def test_steering_penalty_reduced_to_0_02():
    # Hard turn (action 0): penalty = |0-5| * 0.02 = 0.1
    r_straight, _ = compute_reward(_clear_scan(), action_index=5)
    r_turn, _     = compute_reward(_clear_scan(), action_index=0)
    assert r_straight - r_turn == pytest.approx(0.1, abs=1e-4)
```

- [ ] **Step 1c: Verifica che i test falliscano (RED)**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/test_usv_logic.py -v 2>&1"
```

Expected: almeno 7 FAILED (test aggiornati + nuovi). Se tutti passano, i test sono sbagliati — ricontrolla.

---

### Step 2 — Implementa le modifiche in `usv_logic.py`

- [ ] **Step 2a: Sostituisci il contenuto di `src/my_usv/scripts/usv_logic.py`**

```python
import numpy as np

LIDAR_MAX_RANGE    = 5.0
LIDAR_BEAMS        = 50
COLLISION_DIST     = 0.25
FRONT_DANGER       = 3.0    # esteso da 1.5: robot vede muro 15 step prima
SIDE_DANGER        = 0.45
LINEAR_VEL         = 0.5
SPACE_BONUS_WEIGHT = 2.0    # bonus max per spazio aperto


def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    chunks = np.array_split(scan, n_bins)
    return np.array([np.min(chunk) for chunk in chunks])


def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    # FOV 270° / 50 bin = 5.4°/bin → destra [-135°,-54°], fronte [-54°,+54°], sinistra [+54°,+135°]
    right_dist = float(np.min(scan[0:15]))
    front_dist = float(np.min(scan[15:35]))
    left_dist  = float(np.min(scan[35:50]))

    if min(right_dist, front_dist, left_dist) < COLLISION_DIST:
        return -1000.0, True

    # Open-space bonus: incentiva navigazione lontano dai muri
    space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE

    # Steering: penalità ridotta (anti-oscillazione, non vincolo comportamentale)
    steering_penalty = abs(action_index - 5) * 0.02

    danger_penalty = 0.0

    # Front danger quadratico su zona estesa: segnale più forte a distanza media
    if front_dist < FRONT_DANGER:
        severity = (FRONT_DANGER - front_dist) / (FRONT_DANGER - COLLISION_DIST)
        danger_penalty += 20.0 * (severity ** 2)

    if right_dist < SIDE_DANGER:
        severity = (SIDE_DANGER - right_dist) / (SIDE_DANGER - COLLISION_DIST)
        danger_penalty += 5.0 * (severity ** 2)

    if left_dist < SIDE_DANGER:
        severity = (SIDE_DANGER - left_dist) / (SIDE_DANGER - COLLISION_DIST)
        danger_penalty += 5.0 * (severity ** 2)

    return 5.0 + space_bonus - steering_penalty - danger_penalty, False
```

- [ ] **Step 2b: Verifica GREEN**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/test_usv_logic.py -v 2>&1"
```

Expected: tutti PASSED. Se fallisce qualcosa, correggi `usv_logic.py` senza toccare i test.

- [ ] **Step 2c: Verifica intera suite**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v 2>&1"
```

Expected: 39+ test PASSED, 0 FAILED.

- [ ] **Step 2d: Commit**

```bash
git add src/my_usv/scripts/usv_logic.py src/my_usv/test/test_usv_logic.py
git commit -m "feat(reward): space bonus, quadratic front danger, extend FRONT_DANGER to 3m"
```

---

## Task 2: MAX_STEPS 500 → 1000

**Files:**
- Modify: `src/my_usv/scripts/train.py:40`

- [ ] **Step 1: Cambia costante**

In `src/my_usv/scripts/train.py` riga 40:
```python
MAX_STEPS = 1000
```

- [ ] **Step 2: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "feat(train): raise MAX_STEPS from 500 to 1000"
```

---

## Task 3: Phase detection in train.py

**Files:**
- Modify: `src/my_usv/scripts/train.py`

Il training script deve:
1. Accettare `--phase-file` (path a `phase.txt`)
2. Tenere una finestra mobile dei reward per maze 1 (ultimi 50 episodi)
3. Quando `mean(finestra_maze1) > 1500` → scrivere `"2"` in `phase.txt`

- [ ] **Step 1: Aggiungi argomento CLI e costante**

In `parse_args()`, aggiungi dopo `--checkpoint`:
```python
p.add_argument('--phase-file', type=str,
               default='src/my_usv/scripts/phase.txt')
```

Aggiungi costante dopo `MAX_STEPS = 1000`:
```python
PHASE2_THRESHOLD  = 1500   # avg reward maze1 su finestra 50 ep per passare a Phase 2
PHASE1_WINDOW     = 50     # dimensione finestra per calcolo threshold
```

- [ ] **Step 2: Aggiungi funzione di scrittura phase file**

Aggiungi dopo `parse_args()`:
```python
def _write_phase(phase_file: str, phase: int) -> None:
    with open(phase_file, 'w') as f:
        f.write(str(phase))
```

- [ ] **Step 3: Integra rilevamento threshold nel loop principale**

In `main()`, dopo `rh = deque(maxlen=100)` aggiungi:
```python
maze1_window = deque(maxlen=PHASE1_WINDOW)
```

Nel loop episodico, DOPO `rh.append(ep_rew)` e PRIMA del logging CSV, aggiungi:
```python
# Phase detection: monitora maze 1 per transizione Phase 1 → Phase 2
if args.maze_id == 1:
    maze1_window.append(ep_rew)
    if (len(maze1_window) == PHASE1_WINDOW
            and float(np.mean(maze1_window)) > PHASE2_THRESHOLD):
        phase_path = os.path.abspath(args.phase_file)
        if not os.path.exists(phase_path) or open(phase_path).read().strip() == '1':
            _write_phase(phase_path, 2)
            print(f"  🎓 PHASE 2 sbloccata! avg50_maze1={float(np.mean(maze1_window)):.1f} > {PHASE2_THRESHOLD}")
```

- [ ] **Step 4: Verifica sintassi**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && source install/setup.bash && python3 src/my_usv/scripts/train.py --help 2>&1"
```

Expected: help con `--phase-file` visibile, nessun errore di import.

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "feat(train): add phase detection, write phase.txt when maze1 avg50 > 1500"
```

---

## Task 4: Curriculum progressivo in start_training_curriculum.sh

**Files:**
- Modify: `start_training_curriculum.sh`

Logica nuova:
- `PHASE_FILE` = `src/my_usv/scripts/phase.txt`
- Funzione `select_maze()`: legge `PHASE_FILE`, se "2" → random 30/70, altrimenti maze 1
- Rimuovi `MAZE_SEQUENCE` e la logica `block % NUM_MAZES`
- `--phase-file` passato a `train.py`

- [ ] **Step 1: Sostituisci sezione CONFIGURAZIONE**

Rimuovi le righe:
```bash
MAZE_SEQUENCE=(1 2)
```
e
```bash
NUM_MAZES=${#MAZE_SEQUENCE[@]}
TOTAL_BLOCKS=$(( (TOTAL_EPISODES + EPISODES_PER_BLOCK - 1) / EPISODES_PER_BLOCK ))
```

Aggiungi dopo `STATE_FILE`:
```bash
PHASE_FILE="${SCRIPTS_HOST}/phase.txt"
PHASE2_PROB=70   # % probabilità maze 2 in Phase 2 (complementare = 30% maze 1)
```

Aggiungi `TOTAL_BLOCKS` con calcolo indipendente:
```bash
TOTAL_BLOCKS=$(( (TOTAL_EPISODES + EPISODES_PER_BLOCK - 1) / EPISODES_PER_BLOCK ))
```

- [ ] **Step 2: Aggiungi funzione `select_maze()`**

Inserisci dopo la funzione `stop_container`:
```bash
select_maze() {
    local phase=1
    if [ -f "$PHASE_FILE" ]; then
        phase=$(cat "$PHASE_FILE" | tr -d '[:space:]')
    fi

    if [ "$phase" = "2" ]; then
        # Phase 2: 30% maze 1, 70% maze 2
        local roll=$(( RANDOM % 100 ))
        if [ "$roll" -lt "$PHASE2_PROB" ]; then
            echo 2
        else
            echo 1
        fi
    else
        # Phase 1: sempre maze 1
        echo 1
    fi
}
```

- [ ] **Step 3: Aggiorna il loop principale**

Nel loop `for (( block=START_BLOCK; block<TOTAL_BLOCKS; block++ ))`, sostituisci le righe:
```bash
MAZE_IDX=$(( block % NUM_MAZES ))
MAZE_ID=${MAZE_SEQUENCE[$MAZE_IDX]}
```

Con:
```bash
MAZE_ID=$(select_maze)
```

- [ ] **Step 4: Aggiungi `--phase-file` al comando `run_train_block`**

In `run_train_block()`, aggiungi `--phase-file` al comando `python3 train.py`:
```bash
python3 ${SCRIPTS_CTR}/train.py \
    --start-ep   ${start_ep} \
    --end-ep     ${end_ep} \
    --maze-id    ${maze_id} \
    --checkpoint ${CHECKPOINT_CTR} \
    --phase-file ${SCRIPTS_CTR}/phase.txt
```

- [ ] **Step 5: Aggiorna `--reset` per cancellare anche `phase.txt`**

Nel blocco `if [ "$1" = "--reset" ]`, aggiungi `phase.txt` ai file rimossi:
```bash
rm -f "${SCRIPTS_HOST}/checkpoint.pkl" \
      "${SCRIPTS_HOST}/checkpoint.pkl.tmp" \
      "${SCRIPTS_HOST}/phase.txt" \
      "${STATE_FILE}"
```

- [ ] **Step 6: Aggiorna header stampato a console**

Sostituisci la riga:
```bash
printf "║  Epsilon decay      : %-40s║\n" "0.988/ep → ε=0.30 dopo 100 ep"
```
Con:
```bash
printf "║  Curriculum         : %-40s║\n" "Phase1=maze1 | Phase2=30/70 (thr:avg50>1500)"
printf "║  MAX_STEPS          : %-40s║\n" "1000"
```

- [ ] **Step 7: Verifica sintassi bash**

```bash
bash -n start_training_curriculum.sh && echo "Syntax OK"
```

Expected: `Syntax OK` senza errori.

- [ ] **Step 8: Commit**

```bash
git add start_training_curriculum.sh
git commit -m "feat(curriculum): progressive phase1→phase2 with 30/70 maze split"
```

---

## Task 5: Push finale

- [ ] **Step 1: Verifica intera suite test ancora verde**

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v 2>&1"
```

Expected: tutti PASSED.

- [ ] **Step 2: Push**

```bash
git push origin prova_claude_code
```

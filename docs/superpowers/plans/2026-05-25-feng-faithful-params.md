# Feng-Faithful Params Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rendere l'agente DDQN del branch `paper_metric_feng` una replica fedele di Feng et al. 2021 (stato 50 [0,5], reward puro +5/−1000, no frame-stack/heading/domain-rand/grad-clip, 3000 ep, training random su M2), mantenendo invariata la nostra eval harness.

**Architecture:** Modifiche hardcoded ai file core (`ddqn_model.py`, `usv_logic.py`, `usv_env.py`, `train_core.py`) + nuovo orchestratore `start_train_feng.sh` (single-maze M2, 3000 ep) adattato da `start_train_multimaze.sh`. Eval (`test.py`, `aggregate_seeds.py`) intatta. Branch dedicato: r_alpha resta su `paper_metric_base`.

**Tech Stack:** Python 3, PyTorch (DDQN), NumPy, ROS 2 Humble + Gazebo (in Docker), pytest. Test unit eseguiti nel container.

**Spec:** `docs/superpowers/specs/2026-05-25-feng-faithful-params-design.md`
**Branch:** `paper_metric_feng` (già attivo, da `paper_metric_base @ c5b8a06`).
**Cornice:** baseline/àncora di varianza, NON "prova che Feng fallisce" (vedi spec §1 + `DOCUMENTAZIONE/PAPER_ANALYSIS/letteratura_drl_collision_avoidance.md` §0-bis).

---

## File Structure

| File | Responsabilità | Azione |
|---|---|---|
| `src/my_usv/scripts/ddqn_model.py` | Rete + STATE_DIM | Modifica (152→50) |
| `src/my_usv/scripts/usv_logic.py` | LIDAR preprocessing + reward (logica pura) | Modifica (uniform select + reward puro) |
| `src/my_usv/scripts/usv_env.py` | Nodo ROS env: stato, frame buffer, DR | Modifica (FRAME_STACK 1, no heading, no DR, no /5) |
| `src/my_usv/scripts/train_core.py` | Agente DDQN, learn() | Modifica (rimuovi grad-clip) |
| `src/my_usv/test/test_usv_logic.py` | Unit test logica pura | Modifica (rimuovi test shaping+min-pool, attiva test Feng) |
| `start_train_feng.sh` | Orchestratore training M2-only 3000 ep | Crea (da start_train_multimaze.sh) |

**Nessuna modifica a:** `test.py`, `aggregate_seeds.py`, `seeding.py`, `train.py` (gli episodi sono controllati dallo script, non da una costante in train.py), worlds, `paper_metric_base`.

**Comando test (container):**
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/test && python3 -m pytest -v"
```

---

### Task 1: STATE_DIM 152 → 50 (ddqn_model.py)

**Files:**
- Modify: `src/my_usv/scripts/ddqn_model.py:5-7`

- [ ] **Step 1: Modificare STATE_DIM e commento**

Sostituire le righe 5-7:
```python
# Input: 152 dim — 50 raggi × 3 frame stack + cos(yaw) + sin(yaw)
# Output: Q-values per ciascuna delle 11 azioni discrete
STATE_DIM  = 152  # LIDAR_BEAMS * FRAME_STACK + 2
```
con:
```python
# Input: 50 dim — Feng 2021: st = Ot (singola osservazione LIDAR, no frame-stack, no heading)
# Output: Q-values per ciascuna delle 11 azioni discrete
STATE_DIM  = 50  # LIDAR_BEAMS (Feng-faithful)
```

- [ ] **Step 2: Verificare la rete (container)**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/scripts && python3 -c \"import torch; from ddqn_model import DDQN, STATE_DIM; m=DDQN(); assert STATE_DIM==50; assert m.fc1.in_features==50; print(m(torch.zeros(1,50)).shape)\""
```
Expected: stampa `torch.Size([1, 11])`, nessun assert error.

- [ ] **Step 3: Commit**

```bash
git add src/my_usv/scripts/ddqn_model.py
git commit -m "feat(feng): STATE_DIM 152->50 (st=Ot, no frame-stack/heading)"
```

---

### Task 2: process_lidar → selezione uniforme (usv_logic.py)

Feng seleziona 50 misure **uniformemente** dai 512 ray (non min-pool). I test min-pool vanno sostituiti con test di selezione uniforme.

**Files:**
- Modify: `src/my_usv/scripts/usv_logic.py:43-48`
- Test: `src/my_usv/test/test_usv_logic.py:40-52`

- [ ] **Step 1: Sostituire i test min-pool con test uniform-select**

In `test_usv_logic.py`, sostituire le due funzioni `test_min_pooling_picks_nearest_obstacle_in_bin` (righe 40-45) e `test_obstacle_in_last_bin_detected` (righe 48-52) con:
```python
def test_uniform_selection_picks_selected_indices():
    idx = np.linspace(0, 511, 50).round().astype(int)
    raw = [LIDAR_MAX_RANGE] * 512
    raw[idx[3]] = 0.5
    result = process_lidar(raw)
    assert result[3] == pytest.approx(0.5)


def test_uniform_selection_ignores_non_selected_rays():
    idx = np.linspace(0, 511, 50).round().astype(int)
    raw = [LIDAR_MAX_RANGE] * 512
    raw[idx[0] + 1] = 0.5   # ray 1: tra idx[0]=0 e idx[1]≈10, NON selezionato
    result = process_lidar(raw)
    assert result[0] == pytest.approx(LIDAR_MAX_RANGE)


def test_obstacle_in_last_bin_detected():
    raw = [LIDAR_MAX_RANGE] * 512
    raw[-1] = 0.3   # idx[-1]=511 è sempre selezionato
    result = process_lidar(raw)
    assert result[-1] == pytest.approx(0.3)
```

- [ ] **Step 2: Eseguire i nuovi test → devono FALLIRE (process_lidar ancora min-pool)**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/test && python3 -m pytest test_usv_logic.py -k uniform_selection -v"
```
Expected: `test_uniform_selection_ignores_non_selected_rays` FALLISCE (min-pool del bin 0 cattura il ray 1 → result[0]=0.5 invece di MAX_RANGE).

- [ ] **Step 3: Implementare la selezione uniforme**

In `usv_logic.py`, sostituire il corpo di `process_lidar` (righe 43-48):
```python
def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    chunks = np.array_split(scan, n_bins)
    return np.array([np.min(chunk) for chunk in chunks])
```
con:
```python
def process_lidar(raw_ranges, n_bins: int = LIDAR_BEAMS, max_range: float = LIDAR_MAX_RANGE) -> np.ndarray:
    # Feng 2021 §5.1: 50 misure selezionate UNIFORMEMENTE dai 512 ray, clip [0, max_range].
    scan = np.array(raw_ranges, dtype=np.float32)
    scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=max_range)
    scan = np.clip(scan, 0.0, max_range)
    idx = np.linspace(0, len(scan) - 1, n_bins).round().astype(int)
    return scan[idx]
```

- [ ] **Step 4: Eseguire i test process_lidar → tutti verdi**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/test && python3 -m pytest test_usv_logic.py -k 'uniform_selection or output_is_exactly or nan_replaced or pos_inf or values_above or valid_range or last_bin' -v"
```
Expected: tutti PASS (8 test).

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/usv_logic.py src/my_usv/test/test_usv_logic.py
git commit -m "feat(feng): process_lidar selezione uniforme (Feng) al posto del min-pool"
```

---

### Task 3: compute_reward → puro +5 / −1000 (usv_logic.py)

I test reward puro Feng esistono già (`test_usv_logic.py:63-102`) e attualmente FALLISCONO con la reward shaped. I 3 test della shaping R-alpha (righe 109-152) testano comportamento che rimuoviamo → vanno eliminati.

**Files:**
- Modify: `src/my_usv/scripts/usv_logic.py:51-81`
- Test: `src/my_usv/test/test_usv_logic.py:63-152`

- [ ] **Step 1: Eseguire i test reward Feng → confermare che FALLISCONO ora**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/test && python3 -m pytest test_usv_logic.py -k 'returns_exactly_5 or action_index_does_not or far_away_no_collision or exact_collision_boundary' -v"
```
Expected: 4 test FALLISCONO (la reward shaped aggiunge space_bonus/penalty → reward ≠ 5.0).

- [ ] **Step 2: Sostituire compute_reward con la versione pura di Feng**

In `usv_logic.py`, sostituire l'intera `compute_reward` (righe 51-81) con:
```python
def compute_reward(scan: np.ndarray, action_index: int) -> tuple:
    # Feng 2021 Eq.4: reward puro. +5 per step senza collisione, -1000 alla collisione.
    # action_index ignorato (nessuna steering penalty). Le slice settore e gli helper
    # sector_distances/crash_sector restano per il logging/eval, non per la reward.
    if float(np.min(scan)) < COLLISION_DIST:
        return -1000.0, True
    return 5.0, False
```
(Lasciare invariate le costanti FRONT_DANGER, SIDE_DANGER, SPACE_BONUS_WEIGHT, le slice RIGHT/FRONT/LEFT e gli helper: sono usati da `sector_distances`/`crash_sector`/logging.)

- [ ] **Step 3: Rimuovere i test della shaping R-alpha (non più applicabili)**

In `test_usv_logic.py`, eliminare l'intero blocco da riga 105 alla fine del file (commento `# Round 2 (R-alpha)` + le 3 funzioni `test_front_sector_narrow_indices_20_30`, `test_front_penalty_max_weight_is_10`, `test_side_penalty_max_weight_is_3`).

- [ ] **Step 4: Eseguire l'intera suite → tutto verde**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/test && python3 -m pytest test_usv_logic.py -v"
```
Expected: tutti PASS (test process_lidar + test reward Feng), nessun FAIL, nessun test shaping residuo.

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/usv_logic.py src/my_usv/test/test_usv_logic.py
git commit -m "feat(feng): compute_reward puro +5/-1000 (Feng Eq.4), rimossi test shaping R-alpha"
```

---

### Task 4: usv_env → stato 50 [0,5], no frame-stack/heading/DR

**Files:**
- Modify: `src/my_usv/scripts/usv_env.py:19` (FRAME_STACK)
- Modify: `src/my_usv/scripts/usv_env.py:231-241` (blocco DR noise in step_action)
- Modify: `src/my_usv/scripts/usv_env.py:243-260` (_push_frame + get_state)

- [ ] **Step 1: FRAME_STACK 3 → 1**

Riga 19, sostituire:
```python
FRAME_STACK = 3    # Mnih et al. 2015: k frame stack risolve POMDP aliasing
```
con:
```python
FRAME_STACK = 1    # Feng 2021: st = Ot (nessun frame stacking)
```

- [ ] **Step 2: Rimuovere il domain-randomization noise in step_action**

Righe 231-237, sostituire:
```python
        # State su scan NOISY solo in training (perception robustness — Tobin 2017).
        # self.current_scan NON viene modificato: solo lo state path vede il rumore.
        if training:
            noise = np.random.normal(0, DR_NOISE_STD, LIDAR_BEAMS).astype(np.float32)
            scan_for_state = np.clip(self.current_scan + noise, 0.0, LIDAR_MAX_RANGE)
        else:
            scan_for_state = self.current_scan
```
con:
```python
        # Feng 2021: nessun domain randomization. Lo stato è l'osservazione pulita.
        scan_for_state = self.current_scan
```

- [ ] **Step 3: _push_frame senza normalizzazione (feed [0,5])**

Righe 247-251, sostituire:
```python
        normalized = (scan / LIDAR_MAX_RANGE).copy()
        self._frame_buffer.append(normalized)
        # Padding: primi FRAME_STACK-1 step dell'episodio hanno frame iniziale duplicato (Mnih 2015).
        while len(self._frame_buffer) < FRAME_STACK:
            self._frame_buffer.appendleft(normalized)
```
con:
```python
        # Feng 2021: input rete = osservazione clippata [0, 5.0] m, NON normalizzata a [0,1].
        self._frame_buffer.append(scan.copy())
        while len(self._frame_buffer) < FRAME_STACK:
            self._frame_buffer.appendleft(scan.copy())
```

- [ ] **Step 4: get_state ritorna il singolo scan (50), no heading**

Righe 253-260, sostituire:
```python
    def get_state(self) -> np.ndarray:
        """Lettura pura dello stato corrente — non modifica il frame buffer."""
        if not self._frame_buffer:
            raise RuntimeError("get_state() chiamato prima di reset_environment()")
        stacked = np.concatenate(list(self._frame_buffer))                        # 150 dim
        heading = np.array([np.cos(self._current_yaw),
                            np.sin(self._current_yaw)], dtype=np.float32)         # 2 dim
        return np.concatenate([stacked, heading])                                  # 152 dim
```
con:
```python
    def get_state(self) -> np.ndarray:
        """Feng 2021: stato = ultima osservazione LIDAR (50 dim, [0,5] m). No frame-stack, no heading."""
        if not self._frame_buffer:
            raise RuntimeError("get_state() chiamato prima di reset_environment()")
        return self._frame_buffer[-1].copy()   # 50 dim
```

- [ ] **Step 5: Verificare import + coerenza dimensioni (container, no Gazebo)**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/scripts && python3 -c \"import usv_env; from ddqn_model import STATE_DIM; assert usv_env.FRAME_STACK==1; assert STATE_DIM==50; print('FRAME_STACK', usv_env.FRAME_STACK, 'STATE_DIM', STATE_DIM)\""
```
Expected: stampa `FRAME_STACK 1 STATE_DIM 50`, nessun assert error. (Il check funzionale completo di `get_state().shape==(50,)` avviene allo smoke training, Task 7, perché richiede Gazebo.)

- [ ] **Step 6: Commit**

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "feat(feng): stato 50 [0,5], FRAME_STACK=1, no heading, no domain-rand"
```

---

### Task 5: train_core → rimuovere grad-clip

⚠️ Watch-item stabilità (spec §5.3): MSE + spike −1000 senza clip può divergere. Se lo smoke training diverge (loss NaN, Q→∞), questo è il primo sospetto.

**Files:**
- Modify: `src/my_usv/scripts/train_core.py:82-86`

- [ ] **Step 1: Rimuovere clip_grad_norm**

Righe 82-86, sostituire:
```python
        loss = self.loss_fn(self.q_net(s).gather(1, a), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
```
con:
```python
        # Feng 2021: nessun gradient clipping menzionato.
        loss = self.loss_fn(self.q_net(s).gather(1, a), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

- [ ] **Step 2: Verificare che il modulo importi e l'agente si costruisca (container)**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/scripts && python3 -c \"import train_core; a=train_core.DDQNAgent(); assert a.q_net.fc1.in_features==50; print('agent ok, MSE:', type(a.loss_fn).__name__)\""
```
Expected: stampa `agent ok, MSE: MSELoss`, nessun assert error.

- [ ] **Step 3: Commit**

```bash
git add src/my_usv/scripts/train_core.py
git commit -m "feat(feng): rimosso grad-clip (non presente in Feng)"
```

---

### Task 6: start_train_feng.sh (orchestratore M2-only, 3000 ep)

Adattato da `start_train_multimaze.sh`: stessa infra (--seed/--config/runs layout/backup-guard/block-lifecycle/run_meta), ma **solo Maze 2** e **3000 ep** (15 blocchi × 200). Spawn random per-episodio già nativo nell'env (`random.choice(SPAWN_LISTS[2])`).

**Files:**
- Create: `start_train_feng.sh`

- [ ] **Step 1: Creare start_train_feng.sh**

```bash
#!/bin/bash
# =============================================================================
#  start_train_feng.sh  —  Replica fedele Feng et al. 2021 (baseline)
#
#  Training DIRETTO su Maze 2 (no curriculum, no multi-maze), 3000 episodi,
#  spawn random per-episodio. Agente Feng-puro (stato 50 [0,5], reward +5/-1000,
#  no frame-stack/heading/DR/grad-clip). Gazebo riavvia ogni blocco.
#  Eval invariata: usare ./start_test.sh --config=feng --seed=N.
#
#  Uso:
#    ./start_train_feng.sh --seed=0 --config=feng           # riprende
#    ./start_train_feng.sh --seed=0 --config=feng --reset   # backup + riparte
#
#  PREREQUISITO: colcon build eseguito almeno una volta.
# =============================================================================

GAZEBO_SPEED=5
GAZEBO_WAIT=30
TOTAL_BLOCKS=15      # 3000 ep = 15 × 200 (Feng: 3000 epoch)
BLOCK_SIZE=200
MAZE_ID=2            # Feng: training su una sola mappa complessa (≈ Map 2)

SEED=0
CONFIG="feng"
DO_RESET=0
for arg in "$@"; do
    case "$arg" in
        --reset)    DO_RESET=1 ;;
        --seed=*)   SEED="${arg#*=}" ;;
        --config=*) CONFIG="${arg#*=}" ;;
    esac
done
RUN_DIR="runs/${CONFIG}/seed_${SEED}"
mkdir -p "$(pwd)/${RUN_DIR}"

WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
SPAWN_ARGS="x:=-6 y:=0 yaw:=0"   # spawn di launch iniziale; per-episodio è random nell'env

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="/home/usv_ws/${RUN_DIR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))

mkdir -p "$(pwd)/logs"

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
          "${RUN_DIR}/best_model.pth"
    echo "  Reset completato."
fi

echo ""
echo "============================================================"
echo "  USV DDQN — REPLICA FEDELE FENG 2021 (baseline)"
echo "============================================================"
echo "  Maze         : M2 only (random spawn per-episodio)"
echo "  Episodi tot  : ${TOTAL_EP}"
echo "  Blocchi      : ${TOTAL_BLOCKS} x ${BLOCK_SIZE} ep"
echo "  Gazebo speed : ${GAZEBO_SPEED}x headless"
echo "  Seed/Config  : ${SEED} / ${CONFIG}"
echo "  Checkpoint   : ${RUN_DIR}/checkpoint.pkl"
echo "============================================================"
echo ""

trap 'echo ""; echo "  Interruzione ricevuta. Pulizia container..."; docker rm -f usv_container 2>/dev/null; exit 1' INT TERM

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    echo ""
    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Maze ${MAZE_ID} | ep $((START_EP+1))-${END_EP}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container 2>/dev/null
    sleep 1

    LOG_FILE="$(pwd)/logs/feng_block_$(printf '%02d' $b)_maze_${MAZE_ID}.log"

    docker run -d --name usv_container \
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
                --checkpoint ${CHECKPOINT_CTR} \
                --seed       ${SEED}
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
echo "  TRAINING COMPLETATO — ${TOTAL_EP} episodi (Feng baseline)"
echo "============================================================"
echo "  Modello: ${RUN_DIR}/best_model.pth"
echo "  Log:     ${RUN_DIR}/training_log.csv"
echo "============================================================"
```

- [ ] **Step 2: Verificare la sintassi bash**

Run:
```bash
bash -n start_train_feng.sh && echo "Syntax OK"
```
Expected: stampa `Syntax OK`.

- [ ] **Step 3: Commit**

```bash
git add start_train_feng.sh
git commit -m "feat(feng): start_train_feng.sh — orchestratore M2-only 3000 ep con infra rigorosa"
```

---

### Task 7: Validazione finale + smoke training

**Files:**
- Nessuna modifica (solo verifica).

- [ ] **Step 1: Suite unit completa (container)**

Run:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws/src/my_usv/test && python3 -m pytest -v"
```
Expected: tutti PASS. In particolare verdi: i 4 test reward Feng (`returns_exactly_5`, `action_index_does_not_affect_reward`, `far_away_no_collision`, `exact_collision_boundary`) e i test uniform-selection. Nessun test shaping/min-pool residuo.

- [ ] **Step 2: Smoke training 1 blocco (manuale, richiede Docker+Gazebo+X)**

Run un singolo blocco riducendo temporaneamente i blocchi, oppure lanciare e interrompere dopo ~20-30 episodi:
```bash
./start_train_feng.sh --seed=0 --config=feng --reset
# osservare logs/feng_block_01_maze_2.log
```
Expected, nei primi ~30 episodi:
- `get_state()` non solleva errori (stato 50 dim accettato dalla rete 50→300→300→11);
- `avg_loss` nel `runs/feng/seed_0/training_log.csv` è **finito** (no NaN/inf);
- `reward` per episodio coerente con +5/step e −1000 ai crash;
- ε parte ~1.0 e decade come `0.999^k`.
- ⚠️ Se `avg_loss` esplode (NaN/inf) o i Q divergono → re-introdurre `clip_grad_norm_(..., 10.0)` in `train_core.py` (Task 5) e annotare la divergenza come finding (spec §5.3, rischio #1).

- [ ] **Step 3: Aggiornare la memoria di progetto**

Annotare in `project_state.md`: branch `paper_metric_feng` implementato (parametri Feng applicati, `start_train_feng.sh` pronto), watch-item grad-clip, esito smoke test.

- [ ] **Step 4: Commit finale (se Step 2 ha richiesto reintroduzione grad-clip o note)**

```bash
git add -A
git commit -m "test(feng): validazione suite + smoke training 1 blocco"
```

---

## Note di esecuzione

- **NON pushare** senza conferma utente (workflow git di progetto). Commit locali ok.
- I param muti di Feng (γ=0.99, Adam lr=0.00025, batch 64, buffer 100k, target update 5000 step, v=0.5, max-steps 500) restano = r_alpha e vanno **dichiarati come non-da-Feng** in ogni report (già nello spec §2.2).
- Dopo il training: campagna multi-seed (0-4) come per r_alpha, poi `aggregate_seeds.py --config feng`, confronto con r_alpha **sotto stesso protocollo** (baseline, non gotcha).
- `paper_metric_base` (r_alpha) resta intatto: se MSI traina ancora r_alpha, usa quel branch.

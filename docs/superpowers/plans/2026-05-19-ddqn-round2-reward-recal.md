# DDQN Round 2 — Reward Recalibration (R-α) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recalibrare la reward function (`compute_reward`) restringendo il front sector LIDAR a ±27° e dimezzando i pesi delle penalty, poi ri-trainare 5000 episodi multi-maze, testare, e produrre comparazione vs Round 1.

**Architecture:** Modifica isolata di 4 righe in `src/my_usv/scripts/usv_logic.py`, 3 test behavioral nuovi in `src/my_usv/test/test_usv_logic.py`, training reset-from-scratch identico a Round 1 per confronto pulito. Confronto Round 1 vs Round 2 via `analysis_multi_maze_v_19_05.py` (riusato) + writeup `comparison.md`.

**Tech Stack:** Python 3 (numpy, pytest, pytorch), ROS 2 Humble + Gazebo 11 (Docker), bash scripts.

**Branch:** `ddqn_en_20_05` (già creato da `ddqn_en_19_05`).

**Spec:** `docs/superpowers/specs/2026-05-19-ddqn-round2-reward-recal-design.md`

---

## File Structure

| Path | Responsibility | Action |
|---|---|---|
| `src/my_usv/scripts/usv_logic.py` | Reward function + LIDAR processing | Modify (4 lines) |
| `src/my_usv/test/test_usv_logic.py` | Unit tests reward | Modify (add 3 tests) |
| `src/my_usv/scripts/curriculum_state.txt` | Training state | Reset (delete) |
| `src/my_usv/scripts/checkpoint.pkl` | DDQN checkpoint | Reset (delete) |
| `src/my_usv/scripts/training_log.csv` | Training metrics | Reset (delete) |
| `src/my_usv/scripts/test_results.csv` | Test metrics | Reset (delete) |
| `src/my_usv/scripts/best_ddqn_model.pth` | Best model snapshot | Reset (delete) |
| `ANALISI_TRAINING/ANALISI_20_05/` | Post-Round 2 analysis folder | Create |
| `ANALISI_TRAINING/ANALISI_20_05/training_log.csv` | Copy of training log | Copy in (after training) |
| `ANALISI_TRAINING/ANALISI_20_05/test_results.csv` | Copy of test results | Copy in (after testing) |
| `ANALISI_TRAINING/ANALISI_20_05/analysis_multi_maze_v_19_05.py` | Reusable analysis script | Copy from `analisi_maze/` |
| `ANALISI_TRAINING/ANALISI_20_05/plots/` | Generated plots | Create (via script) |
| `ANALISI_TRAINING/ANALISI_20_05/summary_training.txt` | Training summary | Create (via script) |
| `ANALISI_TRAINING/ANALISI_20_05/comparison.md` | R1 vs R2 numeric comparison | Create |

---

## Task 1: Add failing behavioral test — narrow front sector

**Files:**
- Modify: `src/my_usv/test/test_usv_logic.py` (append after line 102)

- [ ] **Step 1: Write failing test for narrow front sector**

Append at end of `src/my_usv/test/test_usv_logic.py`:

```python


# ─────────────────────────────────────────────────────────────────
# Round 2 (R-alpha) — narrow front sector + reduced weights
# ─────────────────────────────────────────────────────────────────

def test_front_sector_narrow_indices_20_30():
    """Bin 15 (era front [15:35], ora right [0:20]) close obstacle:
       front_dist deve NON triggerare. Verifica che front_dist = min(scan[20:30])."""
    scan = _clear_scan()
    scan[15] = 0.5  # close obstacle bin 15 — in nuovo right sector
    reward, done = compute_reward(scan, action_index=5)
    # front_dist = min(scan[20:30]) = LIDAR_MAX_RANGE → no front penalty
    # right_dist = min(scan[0:20]) = 0.5 > SIDE_DANGER(0.45) → no side penalty
    # quindi reward = 5 + space_bonus (nessuna penalty triggered)
    assert done is False
    assert reward > 4.0, f"reward {reward} troppo basso, indica penalty triggered"
```

- [ ] **Step 2: Run test, verify FAIL**

Run (Git Bash, da repo root):
```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest test_usv_logic.py::test_front_sector_narrow_indices_20_30 -v"
```

Expected: FAIL — bin 15 ricade ancora in `front_dist = min(scan[15:35])`, valore 0.5 < 1.5 = FRONT_DANGER triggers, penalty front quadratic = 20 * ((1.5-0.5)/1.25)^2 = 12.8. Reward = 5 + bonus - 12.8 ≈ -6 < 4.0. Test fallisce.

- [ ] **Step 3: Commit failing test**

```bash
git add src/my_usv/test/test_usv_logic.py
git commit -m "test(reward): add failing test for narrow front sector [20:30]

Round 2 R-alpha: bin 15 dovr ricadere in right sector,
non triggerare front_danger phantom."
```

---

## Task 2: Add failing behavioral test — bounded front weight 10

**Files:**
- Modify: `src/my_usv/test/test_usv_logic.py` (append after Task 1 test)

- [ ] **Step 1: Write failing test for bounded front penalty**

Append to `src/my_usv/test/test_usv_logic.py`:

```python


def test_front_penalty_max_weight_is_10():
    """Front bin a 0.26m (just above COLLISION_DIST=0.25) deve produrre
       penalty ≤ 10. Conferma weight ridotto 20→10."""
    scan = _clear_scan()
    # bin 20-29 = front (5.4°/bin × 10 bin = 54° ±27° dall'asse)
    scan[20:30] = 0.26
    reward, done = compute_reward(scan, action_index=5)
    # severity = (1.5 - 0.26) / (1.5 - 0.25) = 0.992
    # penalty front = 10 * 0.992^2 ≈ 9.84
    # mean(scan) = (10*0.26 + 40*5.0)/50 = 4.052
    # bonus = 2.0 * 4.052/5.0 = 1.62
    # reward = 5 + 1.62 - 0 - 9.84 ≈ -3.22
    assert done is False
    assert -4.5 < reward < -1.5, f"reward {reward} fuori range atteso [-4.5, -1.5]"
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest test_usv_logic.py::test_front_penalty_max_weight_is_10 -v"
```

Expected: FAIL — current weight è 20 quindi penalty = 20 * 0.992^2 ≈ 19.68. reward = 5 + 1.62 - 19.68 ≈ -13.06. Fuori range [-4.5, -1.5].

- [ ] **Step 3: Commit failing test**

```bash
git add src/my_usv/test/test_usv_logic.py
git commit -m "test(reward): add failing test for bounded front penalty (weight 10)

Round 2 R-alpha: peso front 20->10."
```

---

## Task 3: Add failing behavioral test — bounded side weight 3

**Files:**
- Modify: `src/my_usv/test/test_usv_logic.py` (append after Task 2 test)

- [ ] **Step 1: Write failing test for bounded side penalty**

Append to `src/my_usv/test/test_usv_logic.py`:

```python


def test_side_penalty_max_weight_is_3():
    """Side bin (bin 0 = right) a 0.26m deve produrre penalty ≤ 3.
       Conferma weight ridotto 5→3."""
    scan = _clear_scan()
    scan[0] = 0.26  # right side close
    reward, done = compute_reward(scan, action_index=5)
    # right_dist = 0.26 < SIDE_DANGER(0.45) → triggers
    # severity = (0.45 - 0.26) / (0.45 - 0.25) = 0.95
    # penalty side = 3 * 0.95^2 ≈ 2.71
    # mean(scan) = (0.26 + 49*5.0)/50 ≈ 4.905
    # bonus = 2.0 * 4.905/5.0 = 1.962
    # reward = 5 + 1.962 - 2.71 ≈ 4.25
    assert done is False
    assert 3.5 < reward < 5.0, f"reward {reward} fuori range atteso [3.5, 5.0]"
```

- [ ] **Step 2: Run test, verify FAIL**

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest test_usv_logic.py::test_side_penalty_max_weight_is_3 -v"
```

Expected: FAIL — current weight è 5 quindi penalty = 5 * 0.95^2 ≈ 4.51. reward = 5 + 1.962 - 4.51 ≈ 2.45. Fuori range [3.5, 5.0].

- [ ] **Step 3: Commit failing test**

```bash
git add src/my_usv/test/test_usv_logic.py
git commit -m "test(reward): add failing test for bounded side penalty (weight 3)

Round 2 R-alpha: peso side 5->3."
```

---

## Task 4: Implement R-α changes in usv_logic.py

**Files:**
- Modify: `src/my_usv/scripts/usv_logic.py` (lines 24-26, 42, 46, 50)

- [ ] **Step 1: Modify sector indices (lines 24-26)**

Open `src/my_usv/scripts/usv_logic.py`. Replace lines 24-26:

**Old:**
```python
    right_dist = float(np.min(scan[0:15]))
    front_dist = float(np.min(scan[15:35]))
    left_dist  = float(np.min(scan[35:50]))
```

**New:**
```python
    right_dist = float(np.min(scan[0:20]))    # 108° destra (R-alpha Round 2: era 81°)
    front_dist = float(np.min(scan[20:30]))   # 54° centro (R-alpha Round 2: era 108°)
    left_dist  = float(np.min(scan[30:50]))   # 108° sinistra (R-alpha Round 2: era 81°)
```

- [ ] **Step 2: Modify front penalty weight (line 42)**

Replace line 42:

**Old:**
```python
        danger_penalty += 20.0 * (severity ** 2)
```

**New:**
```python
        danger_penalty += 10.0 * (severity ** 2)  # R-alpha Round 2: era 20.0
```

- [ ] **Step 3: Modify right side penalty weight (line 46)**

Replace line 46:

**Old:**
```python
        danger_penalty += 5.0 * (severity ** 2)
```

**New:**
```python
        danger_penalty += 3.0 * (severity ** 2)  # R-alpha Round 2: era 5.0
```

- [ ] **Step 4: Modify left side penalty weight (line 50)**

Replace line 50 (è il secondo "danger_penalty += 5.0" nel file):

**Old:**
```python
        danger_penalty += 5.0 * (severity ** 2)
```

**New:**
```python
        danger_penalty += 3.0 * (severity ** 2)  # R-alpha Round 2: era 5.0
```

- [ ] **Step 5: Run the 3 new tests, verify PASS**

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest test_usv_logic.py::test_front_sector_narrow_indices_20_30 test_usv_logic.py::test_front_penalty_max_weight_is_10 test_usv_logic.py::test_side_penalty_max_weight_is_3 -v"
```

Expected: 3 PASSED.

- [ ] **Step 6: Run full test_usv_logic.py to check regressions**

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest test_usv_logic.py -v"
```

Expected:
- PASSED: tutti i test `process_lidar` (7 tests), `test_collision_returns_minus_1000_and_done`, `test_collision_triggered_by_single_ray_below_threshold`, `test_reward_at_exact_collision_boundary`, e i 3 nuovi (3 tests).
- FAILED (pre-broken, expected): `test_no_collision_returns_exactly_5`, `test_action_index_does_not_affect_reward`, `test_obstacle_far_away_no_collision` — già falliscono su Round 1 (assumono Feng pure +5/-1000 ma codice ha space_bonus). NON è regressione Round 2.

Conta finale attesa: ≥13 PASSED, 3 FAILED (pre-existing).

- [ ] **Step 7: Commit implementation**

```bash
git add src/my_usv/scripts/usv_logic.py
git commit -m "feat(reward): R-alpha recalibration — narrow front sector + bounded weights

front_dist: min(scan[20:30]) era min(scan[15:35]) — ±27° invece ±54°.
front penalty weight: 10 era 20.
side penalty weights (right, left): 3 era 5.

Hypothesis: rimuove phantom front_danger in corridoi M1 1.5m
(Pfeiffer 2017), bound penalty magnitude per DDQN stability
(Henderson 2018, Mnih 2015), preserva quadratic gradient (Long 2018)."
```

---

## Task 5: Run full pytest suite to detect cross-file regressions

**Files:**
- No modifications. Read-only validation.

- [ ] **Step 1: Run full pytest suite**

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest -v"
```

Expected:
- `test_usv_logic.py`: 13 PASSED, 3 FAILED (pre-existing, documented in Task 4 Step 6).
- `test_usv_env.py`: tests che dipendevano dal sector NON esistono (sector è usato solo in compute_reward, non in usv_env). Tutti i test devono restare con lo stesso esito di Round 1.
- `test_agent.py` e altri: se presenti, esito invariato.

Tolerance: il numero totale di test FAILED deve restare ≤ ai test FAILED osservati su Round 1 (3 pre-existing in `test_usv_logic.py` + eventuali pre-existing failures in altri file documented in CHANGELOG). Se compaiono NUOVI failures non documentati, fermare e investigare.

- [ ] **Step 2: Documentare baseline test esito**

Crea file temporaneo `pytest_baseline_round2.txt` con output `pytest -v`:

```bash
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c "cd src/my_usv/test && pytest -v 2>&1 | tee /home/usv_ws/pytest_baseline_round2.txt"
```

Verifica `pytest_baseline_round2.txt` esista e contenga il summary atteso.

- [ ] **Step 3: Commit baseline file**

```bash
git add pytest_baseline_round2.txt
git commit -m "test: pytest baseline Round 2 (pre-training)

13 PASSED test_usv_logic.py + 3 pre-existing FAILED documented.
Reference per verificare assenza regressioni durante training."
```

---

## Task 6: Reset training artifacts

**Files:**
- Delete: `src/my_usv/scripts/curriculum_state.txt`
- Delete: `src/my_usv/scripts/checkpoint.pkl`
- Delete: `src/my_usv/scripts/training_log.csv`
- Delete: `src/my_usv/scripts/test_results.csv`
- Delete: `src/my_usv/scripts/best_ddqn_model.pth`

- [ ] **Step 1: Verifica esistenza artifact Round 1**

```bash
ls -la src/my_usv/scripts/curriculum_state.txt src/my_usv/scripts/checkpoint.pkl src/my_usv/scripts/training_log.csv src/my_usv/scripts/test_results.csv src/my_usv/scripts/best_ddqn_model.pth
```

Expected: tutti esistono (sono i risultati Round 1). Se uno manca, è stato già pulito — non è errore.

- [ ] **Step 2: Verifica backup Round 1 in ANALISI_19_05**

```bash
ls ANALISI_TRAINING/ANALISI_19_05/
```

Expected: `summary_training.txt`, `plots/` presenti. Conferma che Round 1 è preservato prima di cancellare i file sorgente.

Se ANALISI_19_05 NON contiene training_log.csv e test_results.csv ma è solo report, copiali ora:

```bash
cp src/my_usv/scripts/training_log.csv ANALISI_TRAINING/ANALISI_19_05/training_log.csv 2>/dev/null || echo "training_log.csv già backup o mancante"
cp src/my_usv/scripts/test_results.csv ANALISI_TRAINING/ANALISI_19_05/test_results.csv 2>/dev/null || echo "test_results.csv già backup o mancante"
```

- [ ] **Step 3: Delete artifact**

```bash
rm -f src/my_usv/scripts/curriculum_state.txt
rm -f src/my_usv/scripts/checkpoint.pkl
rm -f src/my_usv/scripts/training_log.csv
rm -f src/my_usv/scripts/test_results.csv
rm -f src/my_usv/scripts/best_ddqn_model.pth
```

- [ ] **Step 4: Verifica reset**

```bash
ls -la src/my_usv/scripts/checkpoint.pkl src/my_usv/scripts/training_log.csv 2>&1 | grep -E "No such|cannot"
```

Expected: messaggi "No such file or directory" per i file cancellati → reset confermato.

- [ ] **Step 5: Backup Round 1 to ANALISI_TRAINING/ANALISI_19_05 (se non già fatto)**

Se Step 2 ha rilevato che training_log.csv non era in ANALISI_19_05, è stato copiato ora. Verifica:

```bash
ls -la ANALISI_TRAINING/ANALISI_19_05/training_log.csv ANALISI_TRAINING/ANALISI_19_05/test_results.csv
```

Expected: entrambi esistono (sono Round 1 baseline).

NB: questa task NON crea commit (file artifact sono gitignored — confermare con `cat .gitignore | grep -E "checkpoint|training_log|test_results"`).

---

## Task 7: Run training 5000 episodi

**Files:**
- Generate: `src/my_usv/scripts/checkpoint.pkl`
- Generate: `src/my_usv/scripts/training_log.csv`
- Generate: `src/my_usv/scripts/best_ddqn_model.pth`
- Generate: `src/my_usv/scripts/curriculum_state.txt`

- [ ] **Step 1: Avvia training multi-maze**

In Git Bash da repo root:

```bash
./start_train_multimaze.sh
```

Lo script gestisce Gazebo lifecycle per ogni block di 100 ep, alterna M1 e M2 secondo BLOCK_PATTERN=(1,2,2), salva checkpoint ogni 20 ep.

Tempo stimato: 24-36 h wall-clock (GAZEBO_SPEED=4, ~17-25 sec/ep).

Logs in `logs/block_N_maze_M.log`. Training state in `src/my_usv/scripts/curriculum_state.txt`.

- [ ] **Step 2: Monitor progress periodicamente**

Ogni ~6 h verifica:

```bash
tail -20 src/my_usv/scripts/training_log.csv | column -t -s,
cat src/my_usv/scripts/curriculum_state.txt
```

Expected: epoche progrediscono, crash rate diminuisce (verso ~30-50% nei last 100 ep), avg reward last 100 sale.

Red flag: se loss esplode (>1000) o reward last 100 < -2000 stabilmente, abortire (Ctrl+C) e investigare.

- [ ] **Step 3: Attendi completamento 5000 ep**

Training termina automaticamente quando ep_global == 5000. Output finale:
```
✅ Blocco MX completato. avg100=...
```

- [ ] **Step 4: Verifica completezza**

```bash
wc -l src/my_usv/scripts/training_log.csv
tail -1 src/my_usv/scripts/training_log.csv | awk -F, '{print "ep:" $1 " maze:M" $2 " avg100:" $5}'
```

Expected: ~5001 righe (header + 5000 episodi). Ultimo ep_global = 5000.

---

## Task 8: Run test 30 episodi × 3 mazes

**Files:**
- Generate: `src/my_usv/scripts/test_results.csv`

- [ ] **Step 1: Esegui test**

In Git Bash da repo root:

```bash
./start_test.sh
```

Script valuta best_ddqn_model.pth su M1, M2, M3 con 30 ep ciascuno (ε=0.0). Tempo stimato: ~25-35 min wall-clock.

- [ ] **Step 2: Verifica completezza**

```bash
wc -l src/my_usv/scripts/test_results.csv
awk -F, 'NR>1 {print $1}' src/my_usv/scripts/test_results.csv | sort | uniq -c
```

Expected: 91 righe (header + 90 test). 30 ep per ogni maze_id ∈ {1, 2, 3}.

- [ ] **Step 3: Print summary console**

```bash
awk -F, 'NR>1 {crash[$1]+=$5; tot[$1]++; rew[$1]+=$4} END {for (m in tot) printf "M%d: success=%.0f%% avg_reward=%.0f\n", m, (1-crash[m]/tot[m])*100, rew[m]/tot[m]}' src/my_usv/scripts/test_results.csv
```

Expected: 3 righe stampate, una per maze. Confronta con target spec §8: M1 ≥30%, M2 ≥75%.

---

## Task 9: Copy artifacts to ANALISI_TRAINING/ANALISI_20_05/

**Files:**
- Create: `ANALISI_TRAINING/ANALISI_20_05/` (folder)
- Create: `ANALISI_TRAINING/ANALISI_20_05/training_log.csv`
- Create: `ANALISI_TRAINING/ANALISI_20_05/test_results.csv`
- Create: `ANALISI_TRAINING/ANALISI_20_05/analysis_multi_maze_v_19_05.py`
- Create: `ANALISI_TRAINING/ANALISI_20_05/best_ddqn_model.pth`

- [ ] **Step 1: Crea folder**

```bash
mkdir -p ANALISI_TRAINING/ANALISI_20_05
```

- [ ] **Step 2: Copy CSV artifacts**

```bash
cp src/my_usv/scripts/training_log.csv ANALISI_TRAINING/ANALISI_20_05/training_log.csv
cp src/my_usv/scripts/test_results.csv ANALISI_TRAINING/ANALISI_20_05/test_results.csv
cp src/my_usv/scripts/best_ddqn_model.pth ANALISI_TRAINING/ANALISI_20_05/best_ddqn_model.pth
```

- [ ] **Step 3: Copy analysis script**

```bash
cp analisi_maze/analysis_multi_maze_v_19_05.py ANALISI_TRAINING/ANALISI_20_05/analysis_multi_maze_v_19_05.py
```

- [ ] **Step 4: Verifica**

```bash
ls -la ANALISI_TRAINING/ANALISI_20_05/
```

Expected: 4 file presenti.

---

## Task 10: Esegui analysis script e genera plot

**Files:**
- Generate: `ANALISI_TRAINING/ANALISI_20_05/plots/01_reward_curve_global.png` (+ altri plot)
- Generate: `ANALISI_TRAINING/ANALISI_20_05/summary_training.txt`

- [ ] **Step 1: Esegui analysis**

```bash
cd ANALISI_TRAINING/ANALISI_20_05 && python analysis_multi_maze_v_19_05.py && cd ../..
```

Lo script (da `analisi_maze/analysis_multi_maze_v_19_05.py`) legge i CSV dalla stessa cartella, produce 11 plot in `plots/` e `summary_training.txt`.

- [ ] **Step 2: Verifica output**

```bash
ls ANALISI_TRAINING/ANALISI_20_05/plots/
cat ANALISI_TRAINING/ANALISI_20_05/summary_training.txt | head -40
```

Expected:
- 11 plot PNG (01-11).
- `summary_training.txt` con sezione GLOBAL + M1 + M2 + spawn breakdown.

- [ ] **Step 3: Commit analysis output**

```bash
git add ANALISI_TRAINING/ANALISI_20_05/
git commit -m "data(round2): training + test results + plots ANALISI_20_05

Multi-maze 5000 ep training, 30 ep test x 3 mazes.
Best model preservato. Plot via analysis_multi_maze_v_19_05.py."
```

---

## Task 11: Scrivi comparison.md R1 vs R2

**Files:**
- Create: `ANALISI_TRAINING/ANALISI_20_05/comparison.md`

- [ ] **Step 1: Estrai metriche R1 da summary**

Apri `ANALISI_TRAINING/ANALISI_19_05/summary_training.txt` e annota:
- M1: episodes, avg reward last 30, max-steps survival rate, success rate (da test plot 09_test_M1)
- M2: idem (test plot 10_test_M2)
- M3: success rate test (test plot 11_test_M3)

Per Round 1 dati noti (vedi `docs/superpowers/specs/2026-05-19-ddqn-round2-reward-recal-design.md` §8):
- M1 test success: 0% (0/30 P1, 0/30 P2)
- M1 avg reward last 30: -898
- M1 training max-steps survival: 11.4%
- M2 test success: 82% (4 spawn @100%, A1@45%, C2@0%)
- M3 test success: 0%

- [ ] **Step 2: Estrai metriche R2 da nuovo summary**

```bash
cat ANALISI_TRAINING/ANALISI_20_05/summary_training.txt
```

Annota stesse metriche.

Per test success rate per-maze:
```bash
awk -F, 'NR>1 && $1==1 {if ($5==0) succ++; tot++} END {printf "M1: %d/%d = %.1f%%\n", succ, tot, succ/tot*100}' ANALISI_TRAINING/ANALISI_20_05/test_results.csv
awk -F, 'NR>1 && $1==2 {if ($5==0) succ++; tot++} END {printf "M2: %d/%d = %.1f%%\n", succ, tot, succ/tot*100}' ANALISI_TRAINING/ANALISI_20_05/test_results.csv
awk -F, 'NR>1 && $1==3 {if ($5==0) succ++; tot++} END {printf "M3: %d/%d = %.1f%%\n", succ, tot, succ/tot*100}' ANALISI_TRAINING/ANALISI_20_05/test_results.csv
```

- [ ] **Step 3: Crea comparison.md**

Crea `ANALISI_TRAINING/ANALISI_20_05/comparison.md`:

```markdown
# Round 1 vs Round 2 — Comparison

**Round 1 (ddqn_en_19_05):** Multi-maze M1+M2 ratio 1:2 + DR LIDAR σ=0.02 + D1 relocato. Reward original (front sector ±54°, weights front=20 side=5).

**Round 2 (ddqn_en_20_05):** R-α recalibrazione reward — front sector ±27° (bin [20:30]), weights front=10 side=3. Tutto il resto identico.

## Test results (30 ep × 3 mazes, ε=0)

| Metric | Round 1 | Round 2 | Δ | Target met? |
|---|---|---|---|---|
| M1 test success rate | 0% | **<FILL R2 M1>%** | <+ΔM1> | ≥30% target |
| M1 P1 (-2.9,-2.0) success | 0/30 | **<FILL>**/30 | <Δ> | — |
| M1 P2 (1.0,-1.0) success | 0/30 | **<FILL>**/30 | <Δ> | — |
| M2 test success rate | 82% | **<FILL R2 M2>%** | <ΔM2> | ≥75% target |
| M3 test success rate | 0% | **<FILL R2 M3>%** | <ΔM3> | regression check |

## Training metrics

| Metric | Round 1 | Round 2 | Δ |
|---|---|---|---|
| M1 avg reward last 30 ep | -898 | **<FILL>** | <Δ> |
| M1 max-steps survival rate (all) | 11.4% | **<FILL>%** | <Δ> |
| M2 avg reward last 30 ep | +1080 | **<FILL>** | <Δ> |
| M2 max-steps survival rate (all) | 22.7% | **<FILL>%** | <Δ> |
| Global crash rate last 100 | 67% | **<FILL>%** | <Δ> |

## Hypothesis verification

**H1 (narrow front sector elimina phantom):**
- Predizione: M1 reward last 30 ep > +1500.
- Osservato: **<FILL>**.
- Conferma: **<YES/NO/PARTIAL>**.

**H2 (bounded weights stabilizzano DDQN):**
- Predizione: M2 success ≥75% (regressione lieve accettabile).
- Osservato: **<FILL>%**.
- Conferma: **<YES/NO>**.

## Decision tree (spec §8)

Applicare:
- M1 ≥30% & M2 ≥75% → R-α confermato. Round 3 = analizzare M3 / best-model bias.
- M1 ≥30% & M2 <70% → weights troppo ridotti. Round 3 = R-α' weights 15/4.
- M1 <10% → R-α insufficient. Round 3 = R-β (linearize) o investigare ulteriore.
- Tutto regredito → bug. Revert Round 2, debug.

**Risultato decisione:** **<FILL outcome>**

**Round 3 direction:** **<FILL based on decision tree>**
```

Sostituisci ogni `<FILL ...>` con valore numerico estratto da step 1 e 2.

- [ ] **Step 4: Verifica comparison.md**

```bash
cat ANALISI_TRAINING/ANALISI_20_05/comparison.md
```

Verifica: tutti i `<FILL>` sostituiti con numeri reali. Nessun placeholder rimasto.

- [ ] **Step 5: Commit comparison**

```bash
git add ANALISI_TRAINING/ANALISI_20_05/comparison.md
git commit -m "docs(round2): comparison R1 vs R2 con decision tree applicato

Risultati Round 2 R-alpha vs baseline Round 1.
Decision tree applicato per Round 3 direction."
```

---

## Task 12: Update memory + briefing

**Files:**
- Modify: `C:\Users\david\.claude\projects\C--Users-david-Desktop-PROGETTO-ROBOTICS-Robotics-Collision-Avoidance\memory\project_state.md`
- Modify: `C:\Users\david\.claude\projects\C--Users-david-Desktop-PROGETTO-ROBOTICS-Robotics-Collision-Avoidance\memory\MEMORY.md`
- Create: `DOCUMENTAZIONE/BRIEFING_20_05.md` (se utente lo richiede)

- [ ] **Step 1: Update memory project_state.md**

Apri il file e sostituisci la sezione corrente con:

```markdown
---
name: project-state
description: Current project state — Round 2 R-alpha completato 2026-05-XX. Risultati test M1=<R2 M1%>, M2=<R2 M2%>, M3=<R2 M3%>. Round 1 baseline preservato ANALISI_19_05. Branch ddqn_en_20_05 pushed.
metadata:
  type: project
---

Round 2 (ddqn_en_20_05) IMPLEMENTATO + pushato 2026-05-XX.
R-α reward recalibration: narrow front sector + bounded weights.

**Why:** Round 1 produsse M2=82% best-ever ma M1=0% per phantom front_danger in corridoi 1.5m (analisi geometrica). R-α corregge la geometria del sector e bounda i weights.

**How to apply:** Round 3 direction = vedi `ANALISI_TRAINING/ANALISI_20_05/comparison.md` decision tree.

Round 1 best: M2=82%, M1=0%, M3=0% (ANALISI_19_05).
Round 2 risultati: <FILL post-test> (ANALISI_20_05).
```

- [ ] **Step 2: Update MEMORY.md index**

Cambia la riga relativa a project_state.md per riflettere il nuovo stato:

```markdown
- [project_state.md](project_state.md) — Round 2 R-alpha completato 2026-05-XX. M1=<X>%, M2=<Y>%, M3=<Z>%. Branch ddqn_en_20_05.
```

- [ ] **Step 3: Verifica memory**

Lettura per controllo:

```bash
cat "C:/Users/david/.claude/projects/C--Users-david-Desktop-PROGETTO-ROBOTICS-Robotics-Collision-Avoidance/memory/project_state.md"
cat "C:/Users/david/.claude/projects/C--Users-david-Desktop-PROGETTO-ROBOTICS-Robotics-Collision-Avoidance/memory/MEMORY.md"
```

Expected: project_state.md aggiornato con risultati R2. MEMORY.md index aggiornato.

---

## Task 13: Push branch su GitHub

**Files:** No file modificati, operazione git remote.

- [ ] **Step 1: Verifica clean tree**

```bash
git status
```

Expected: clean working tree, branch `ddqn_en_20_05` con commits Task 1-11.

- [ ] **Step 2: Push branch**

```bash
git push -u origin ddqn_en_20_05
```

Expected: branch creato su remote, tracking set up.

- [ ] **Step 3: Verifica remote**

```bash
git branch -vv | grep ddqn_en_20_05
```

Expected: linea contiene `[origin/ddqn_en_20_05]`.

---

## Self-Review

**Spec coverage check** vs `docs/superpowers/specs/2026-05-19-ddqn-round2-reward-recal-design.md`:

| Spec §  | Requirement | Task |
|---|---|---|
| §5.1 (4 righe usv_logic.py) | Sector + weights | Task 4 |
| §5.2 (3 nuovi test) | Behavioral tests | Task 1, 2, 3 |
| §5.2 (test esistenti che restano) | No regression | Task 4 Step 6, Task 5 |
| §5.3 (nessun cambio altrove) | Isolation | Task 4 (solo usv_logic.py) |
| §6 (reset checkpoint + training 5000 ep) | Training | Task 6, 7 |
| §7 (30 ep × 3 mazes test) | Testing | Task 8 |
| §8 (analysis + comparison 4 metriche) | Evaluation | Task 9, 10, 11 |
| §8 (decision tree) | Round 3 direction | Task 11 Step 3 |
| §10 (rollback path) | Isolated branch | Task 13 (push), branch già creato |
| §11 acceptance criteria | All covered | Task 1-13 |

Tutte le sezioni coperte.

**Placeholder scan:** Nessun TBD/TODO. I `<FILL>` in comparison.md (Task 11) sono **markers per valori numerici da popolare in fase di esecuzione**, non placeholder di plan — il task istruisce esplicitamente cosa sostituirli. Stessa logica per Task 12.

**Type consistency:** Nomi funzioni/variabili coerenti — `compute_reward`, `_clear_scan`, `front_dist`, `right_dist`, `left_dist`, `severity`, `danger_penalty`, `FRONT_DANGER`, `SIDE_DANGER`, `COLLISION_DIST`, `LIDAR_BEAMS`, `LIDAR_MAX_RANGE` — match esatto al codice corrente di `usv_logic.py`.

---

## Execution choice

**Plan complete and saved to `docs/superpowers/plans/2026-05-19-ddqn-round2-reward-recal.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — dispatch fresh subagent per task con review tra task, iterazione veloce.

**2. Inline Execution** — esecuzione tasks in questa sessione con executing-plans, batch con checkpoint.

Note: Task 7 (training 5000 ep, 24-36h) e Task 8 (test 30 ep) richiedono Docker + Gazebo running su Windows host. Subagenti possono fare Task 1-6 (code + tests), poi handoff a te per Task 7-13. Inline è simile.

**Which approach?**

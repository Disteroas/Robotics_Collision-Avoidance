# merge15_05 Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare branch merge15_05 for USV DDQN training — 8000 ep M2-only, 7 spawn points (3 removed), fresh start.

**Architecture:** Three config-only changes (spawn list, block count, test episodes) + docs update. No algorithmic changes. Branch from merge14_05 to inherit analysis_v_15_05.py, test.py spawn logging, and analysis script updates.

**Tech Stack:** Python 3, ROS 2, Gazebo, Docker. Training via `./start_train_multimaze.sh --reset`.

---

### Task 1: Create branch merge15_05

**Files:**
- No file changes — git operation only.

- [ ] **Step 1: Verify current branch is merge14_05**

```bash
git branch --show-current
```
Expected output: `merge14_05`

- [ ] **Step 2: Create and switch to merge15_05**

```bash
git checkout -b merge15_05
```
Expected output: `Switched to a new branch 'merge15_05'`

- [ ] **Step 3: Verify**

```bash
git branch --show-current
```
Expected output: `merge15_05`

---

### Task 2: Remove 3 toxic spawn points from SPAWN_LISTS[2]

**Files:**
- Modify: `src/my_usv/scripts/usv_env.py` lines 20–42

**Context:** `SPAWN_LISTS[2]` currently contains 10 M2 training spawn points. Three must be removed based on 3-run empirical analysis (all had 0% completion and avg steps < 130):
- D2 `(0.5, -2.0, 1.571)` — avg 110 steps, 0% completion
- D3 `(3.5, 0.5, 4.712)` — avg 55 steps, 0% completion (catastrophic)
- E2 `(0.0, 3.5, 3.142)` — avg 128 steps, 0% completion

`TEST_SPAWN_LISTS[2]` is a separate list — leave it unchanged. D2 and E2 appear there and must remain for test reproducibility.

- [ ] **Step 1: Open usv_env.py and locate SPAWN_LISTS[2]**

The section to replace is lines 20–42. Current state:

```python
    2: [
        # Zone A: ingresso sinistro (1 spawn) — A2 rimosso (rischio uscita labirinto)
        (-6.0,  0.0,  0.0  ),  # A1: heading E  — min=1.352m

        # Zone B: centro-sinistra (1 spawn) — B1/B2 rimossi (percorso interno, U-turn non verificato)
        (-4.5,  1.5,  2.356),  # B3: heading NW — min=0.497m

        # Zone C: centro (1 spawn) — C1/C3 rimossi (percorso interno, U-turn non verificato)
        (-7.0,  5.0,  0.0  ),  # C2: heading E  — min=0.860m

        # Zone D: centro-destra (3 spawn) — validated min≥0.43m
        ( 1.5,  0.0,  3.142),  # D1: heading W  — min=0.693m
        ( 0.5, -2.0,  1.571),  # D2: heading N  — min=0.430m
        ( 3.5,  0.5,  4.712),  # D3: heading S  — min=0.780m

        # Zone E: superiore (1 spawn) — E1 rimosso (percorso interno, U-turn non verificato)
        ( 0.0,  3.5,  3.142),  # E2: heading W  — min=0.650m

        # Zone F: inferiore (3 spawn) — validated min≥0.43m
        (-4.5, -3.5,  0.0  ),  # F1: heading E  — min=1.162m
        (-1.5, -4.0,  1.571),  # F2: heading N  — min=1.008m
        ( 6.0,  6.0,  3.142),  # F3: heading W  — min=0.456m
    ],
```

- [ ] **Step 2: Replace with 7-spawn version**

Replace the entire block above with:

```python
    2: [
        # Zone A: ingresso sinistro (1 spawn) — A2 rimosso (rischio uscita labirinto)
        (-6.0,  0.0,  0.0  ),  # A1: heading E  — min=1.352m

        # Zone B: centro-sinistra (1 spawn) — B1/B2 rimossi (percorso interno, U-turn non verificato)
        (-4.5,  1.5,  2.356),  # B3: heading NW — min=0.497m

        # Zone C: centro (1 spawn) — C1/C3 rimossi (percorso interno, U-turn non verificato)
        (-7.0,  5.0,  0.0  ),  # C2: heading E  — min=0.860m

        # Zone D: centro-destra (1 spawn) — D2/D3 rimossi (0% completion su 3 run, <130 step medi)
        ( 1.5,  0.0,  3.142),  # D1: heading W  — min=0.693m

        # Zone F: inferiore (3 spawn) — validated min≥0.43m
        (-4.5, -3.5,  0.0  ),  # F1: heading E  — min=1.162m
        (-1.5, -4.0,  1.571),  # F2: heading N  — min=1.008m
        ( 6.0,  6.0,  3.142),  # F3: heading W  — min=0.456m
    ],
```

Note: Zone E section is completely removed (E2 was its only entry).

- [ ] **Step 3: Verify spawn count**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src/my_usv/scripts')
from usv_env import SPAWN_LISTS
print('M2 training spawns:', len(SPAWN_LISTS[2]))
for s in SPAWN_LISTS[2]:
    print(' ', s)
"
```

Expected output:
```
M2 training spawns: 7
  (-6.0, 0.0, 0.0)
  (-4.5, 1.5, 2.356)
  (-7.0, 5.0, 0.0)
  (1.5, 0.0, 3.142)
  (-4.5, -3.5, 0.0)
  (-1.5, -4.0, 1.571)
  (6.0, 6.0, 3.142)
```

- [ ] **Step 4: Verify TEST_SPAWN_LISTS[2] is unchanged (still 6 entries, D2 and E2 still present)**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src/my_usv/scripts')
from usv_env import TEST_SPAWN_LISTS
print('M2 test spawns:', len(TEST_SPAWN_LISTS[2]))
for s in TEST_SPAWN_LISTS[2]:
    print(' ', s)
"
```

Expected output:
```
M2 test spawns: 6
  (-6.0, 0.0, 0.0)
  (-4.5, 1.5, 2.356)
  (-7.0, 5.0, 0.0)
  (0.5, -2.0, 1.571)
  (0.0, 3.5, 3.142)
  (-4.5, -3.5, 0.0)
```

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "feat(env): rimuovi D2/D3/E2 da SPAWN_LISTS M2 — 10→7 spawn training"
```

---

### Task 3: Extend training to 8000 episodes

**Files:**
- Modify: `start_train_multimaze.sh` line 18 (`TOTAL_BLOCKS`)

- [ ] **Step 1: Open start_train_multimaze.sh and locate TOTAL_BLOCKS**

Line to change:
```bash
TOTAL_BLOCKS=20
```

- [ ] **Step 2: Change to 40**

```bash
TOTAL_BLOCKS=40
```

`BLOCK_SIZE=200` and `BLOCK_PATTERN=(2)` remain unchanged. `TOTAL_EP` is computed automatically as `TOTAL_BLOCKS * BLOCK_SIZE = 40 * 200 = 8000`.

- [ ] **Step 3: Verify**

```bash
grep -E "TOTAL_BLOCKS|BLOCK_SIZE|TOTAL_EP|BLOCK_PATTERN" start_train_multimaze.sh
```

Expected output:
```
TOTAL_BLOCKS=40
BLOCK_SIZE=200
BLOCK_PATTERN=(2)
...
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))
```

Then verify the computed value:
```bash
bash -c 'TOTAL_BLOCKS=40; BLOCK_SIZE=200; echo "TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))"'
```
Expected: `TOTAL_EP=8000`

- [ ] **Step 4: Commit**

```bash
git add start_train_multimaze.sh
git commit -m "feat(train): TOTAL_BLOCKS 20→40 — 8000 ep training per merge15_05"
```

---

### Task 4: Increase test episodes to 90 per maze

**Files:**
- Modify: `start_test.sh` line 22 (`EPISODES_PER_MAZE`)

**Context:** 3 run analysis showed CI ±18pp with 30 episodes (5 per spawn). 90 episodes (15 per spawn) reduces CI to ±10pp — necessary for statistically meaningful comparisons.

- [ ] **Step 1: Locate EPISODES_PER_MAZE in start_test.sh**

Line to change:
```bash
EPISODES_PER_MAZE=30      # episodi di valutazione per labirinto
```

- [ ] **Step 2: Change to 90**

```bash
EPISODES_PER_MAZE=90      # episodi di valutazione per labirinto (15 per spawn, CI ±10pp)
```

- [ ] **Step 3: Verify**

```bash
grep EPISODES_PER_MAZE start_test.sh
```

Expected: `EPISODES_PER_MAZE=90      # episodi di valutazione per labirinto (15 per spawn, CI ±10pp)`

- [ ] **Step 4: Commit**

```bash
git add start_test.sh
git commit -m "feat(test): EPISODES_PER_MAZE 30→90 — riduce CI da ±18pp a ±10pp"
```

---

### Task 5: Document Esperimento 9 in ESPERIMENTI.md

**Files:**
- Modify: `DOCUMENTAZIONE/ESPERIMENTI.md` — add entry after Esperimento 8

- [ ] **Step 1: Open ESPERIMENTI.md and locate the end of Esperimento 8**

Find the line that ends Esperimento 8:
```
**Piano:** `docs/superpowers/plans/2026-05-14-merge14-training.md`
```

- [ ] **Step 2: Add Esperimento 9 entry after the `---` separator following Esperimento 8**

Append the following block (after the `---` that closes Esperimento 8):

```markdown
## Esperimento 9 — `merge15_05` ← IMPLEMENTATO (training da avviare)

**Branch:** `merge15_05` (da `merge14_05`)
**Data implementazione:** 2026-05-15
**Configurazione:**
- Maze 2 only (`BLOCK_PATTERN=(2)`)
- **8000 ep totali, 40 blocchi × 200 ep**
- BETA_DECAY=0.999 (invariato), MAX_STEPS=500 (invariato)
- **REPLAY_START_SIZE=10,000** (invariato)
- **7 spawn M2 training** (rimossi D2 (0.5,-2.0), D3 (3.5,0.5), E2 (0.0,3.5))
- Ripartenza da zero (checkpoint non riutilizzato)

**Motivazione:**
1. **Non-convergenza merge14_05:** curva avg100 ancora in salita a ep 4000 in run3 (peak@ep3818, final avg100=700). 8000 ep per raggiungere convergenza.
2. **Spawn tossici rimossi:** D2/D3/E2 = 0% completion su ~1180 ep totali (3 run). Avg steps 55-128. Pure gradient noise nel replay buffer.
3. **Ripartenza pulita:** validità scientifica — mix di regime vecchio/nuovo invalida l'analisi comparativa.
4. **Test a 90 ep/maze:** riduce CI da ±18pp a ±10pp per confronti affidabili.

**Avvio training:** `./start_train_multimaze.sh --reset`

**Target:**

| Metrica | Target | Baseline (randomSpawn 05_08) | merge14_05 best (run3) |
|---------|--------|------------------------------|------------------------|
| M2      | ≥ 50%  | 26.7%                        | 30%                    |
| M3      | ≥ 40%  | 40.0%                        | 13%                    |

**Spec:** `docs/superpowers/specs/2026-05-15-merge15-training-design.md`
**Piano:** `docs/superpowers/plans/2026-05-15-merge15-training.md`

---
```

- [ ] **Step 3: Verify the file compiles (no broken markdown)**

```bash
grep -c "## Esperimento" DOCUMENTAZIONE/ESPERIMENTI.md
```
Expected: `9`

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/ESPERIMENTI.md
git commit -m "docs: aggiungi Esperimento 9 (merge15_05) in ESPERIMENTI.md"
```

---

### Task 6: Push branch and final check

- [ ] **Step 1: Verify all changes are committed**

```bash
git status
```
Expected: `nothing to commit, working tree clean`

- [ ] **Step 2: Verify commit log**

```bash
git log --oneline -5
```
Expected (most recent 4 commits on this branch):
```
<sha> docs: aggiungi Esperimento 9 (merge15_05) in ESPERIMENTI.md
<sha> feat(test): EPISODES_PER_MAZE 30→90 — riduce CI da ±18pp a ±10pp
<sha> feat(train): TOTAL_BLOCKS 20→40 — 8000 ep training per merge15_05
<sha> feat(env): rimuovi D2/D3/E2 da SPAWN_LISTS M2 — 10→7 spawn training
```

- [ ] **Step 3: Push branch**

```bash
git push -u origin merge15_05
```

- [ ] **Step 4: Final sanity check — print training config summary**

```bash
bash -c '
  source /dev/null
  TOTAL_BLOCKS=40; BLOCK_SIZE=200
  echo "=== merge15_05 config ==="
  echo "Total episodes : $((TOTAL_BLOCKS * BLOCK_SIZE))"
  echo "Blocks         : ${TOTAL_BLOCKS} x ${BLOCK_SIZE} ep"
  echo "Maze           : M2 only"
  echo "Training spawns: 7 (removed D2, D3, E2)"
  echo "Test episodes  : 90 per maze"
  echo "Start command  : ./start_train_multimaze.sh --reset"
'
```

Expected:
```
=== merge15_05 config ===
Total episodes : 8000
Blocks         : 40 x 200 ep
Maze           : M2 only
Training spawns: 7 (removed D2, D3, E2)
Test episodes  : 90 per maze
Start command  : ./start_train_multimaze.sh --reset
```

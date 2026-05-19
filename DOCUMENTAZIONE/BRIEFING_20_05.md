# Briefing 2026-05-20 — Round 2 R-α pre-training + flag test pre-broken

**Branch precedente:** `ddqn_en_19_05` (training Round 1 completato, M2=82%, M1=0%, M3=0%)
**Branch corrente:** `ddqn_en_20_05`
**Stato:** pre-training (codice implementato, training non ancora avviato)
**Autori:** Davide Covolo (+ Claude)

---

## 1. Modifiche Round 2 R-α (reward recalibration)

Singola variabile isolata: ricalibrare `compute_reward` in `src/my_usv/scripts/usv_logic.py`.

| Componente | Round 1 (era) | Round 2 (è) |
|---|---|---|
| Front sector bin range | `scan[15:35]` (±54°, 108° tot) | `scan[20:30]` (±27°, 54° tot) |
| Right sector | `scan[0:15]` (81°) | `scan[0:20]` (108°) |
| Left sector | `scan[35:50]` (81°) | `scan[30:50]` (108°) |
| Front penalty weight | `20.0 * severity²` | `10.0 * severity²` |
| Side penalty weight (×2) | `5.0 * severity²` | `3.0 * severity²` |
| Max danger penalty totale | 30 | 16 |

Quadratic shape, `FRONT_DANGER=1.5`, `SIDE_DANGER=0.45`, `COLLISION_DIST=0.25`, `SPACE_BONUS_WEIGHT=2.0` invariati.

**Hypothesis primaria (H1):** narrow front sector rimuove phantom front_danger in corridoi M1 1.5m. Letteratura: Pfeiffer et al. 2017.

**Hypothesis secondaria (H2):** bounded weights riducono varianza Q-target nel buffer DDQN. Letteratura: Henderson et al. 2018, Mnih et al. 2015.

**Target:** M1 test ≥30%, M2 test ≥75% (era 82%, accettiamo regressione lieve).

Spec completa: `docs/superpowers/specs/2026-05-19-ddqn-round2-reward-recal-design.md`
Plan: `docs/superpowers/plans/2026-05-19-ddqn-round2-reward-recal.md`

---

## 2. Test pre-broken (flag per Round 3)

Durante implementazione Round 2 emersi **4 test pre-existing fail** in `src/my_usv/test/test_usv_logic.py`. **Nessuno causato da Round 2.** Tutti dovuti a divergenza tra reward function corrente (con `space_bonus` augmentation da merge16_05) e assertion originali (scritte per Feng pure `+5/-1000`).

### Lista test pre-broken

| Test | Riga | Atteso (asserzione) | Reale (codice corrente) | Causa |
|---|---|---|---|---|
| `test_no_collision_returns_exactly_5` | 78 | `reward == 5.0` | `reward == 7.0` | space_bonus=2.0 sommato in scan tutto a max range |
| `test_action_index_does_not_affect_reward` | 84 | tutti i 11 reward == 5.0 | tutti ~7.0 ma diversi tra loro per steering_penalty | steering_penalty 0.02·\|a-5\| introdotta merge16_05 |
| `test_obstacle_far_away_no_collision` | 90 | `reward == 5.0` con scan tutto a 0.26m | reward molto negativo | scan 0.26m → triggera front + 2×side danger penalty quadratic |
| `test_reward_at_exact_collision_boundary` | 97 | `reward == 5.0` con scan tutto a `COLLISION_DIST` | reward ~-25 | stessa ragione del precedente (penalty massicci a d=0.25m) |

### Perché non fixati in Round 2

Spec §5.2: *"Fuori scope Round 2. Non toccare. Flag in BRIEFING_20_05.md per Round 3 fix."*

Round 2 isola **una sola variabile** (reward calibration). Aggiungere fix test = mescolare variabili = perdere capacità di attribuire success/failure a R-α isolato.

### Raccomandazione Round 3

Quando Round 3 verrà progettato, includere come task **separato** (non legato a reward changes):

**Option A — Aggiorna assertion dei 4 test al reward effettivo:**
```python
# test_no_collision_returns_exactly_5 → rinominare e ricalcolare
def test_no_collision_clear_scan_returns_base_plus_bonus():
    reward, done = compute_reward(_clear_scan(), action_index=5)
    # base 5 + space_bonus (2.0 * mean(5.0)/5.0 = 2.0) - 0 - 0 = 7.0
    assert reward == pytest.approx(7.0)
    assert done is False
```

**Option B — Refattorizza compute_reward esponendo le componenti:**
```python
def compute_reward(scan, action_index):
    return _base_reward() + _space_bonus(scan) - _steering_penalty(action_index) - _danger_penalty(scan), _is_collision(scan)
```
Test ognuna componente isolata. Più pulito ma scope grande.

**Raccomandazione:** Option A. YAGNI per refactor.

### Acceptance criterion Round 3

Tutti i `test_usv_logic.py` tests passano (0 FAIL).

---

## 3. Code review notes Round 2

Durante implementazione code review trovato **2 issue Important + 3 Minor**. Applicati:

1. **Important:** `usv_logic.py:9` header comment stale (`right [0:15], front [15:35], left [35:50]`) → aggiornato a `[0:20]/[20:30]/[30:50]`.
2. **Important:** `test_usv_logic.py:119` assert `reward > 4.0` troppo loose (max side penalty 2.7 superava soglia) → tighten a `reward > 6.0`.
3. **Minor:** Commento espanso con valore atteso ~6.96 nel test 1.

Commit: `159b63e fix(review): code review fixes Tasks 1-5`.

---

## 4. Stato corrente

| Task | Stato |
|---|---|
| 1-3 — failing tests TDD | ✅ committed |
| 4 — usv_logic.py impl | ✅ committed |
| 5 — pytest baseline | ✅ committed |
| Code review fixes | ✅ committed |
| 6 — reset artifacts | ✅ no-op (scripts/ già puliti) |
| 7 — Training 5000 ep | ⏳ pending USER (~8h Docker) |
| 8 — Test 90 ep × 3 mazes (270 totali) | ⏳ pending USER |
| 9-13 — analysis + comparison + memory + push | ⏳ post-training |

**Branch:** `ddqn_en_20_05` con 7 commit pre-training. Pronto per `./start_train_multimaze.sh`.

---

## 5. Round 1 raw data NOTA

`ANALISI_TRAINING/ANALISI_19_05/` contiene solo `summary_training.txt` + 11 plot PNG. Raw CSV (`training_log.csv`, `test_results.csv`) **non backup-ati prima del reset** → permanentemente persi.

**Effetto comparison.md (Task 11):** numeri aggregati R1 disponibili da `summary_training.txt` (avg reward last 30, success rate per spawn, ecc.) → comparison.md possibile a livello metric-aggregato. Trajectory-level analysis R1 non possibile.

**Lezione Round 3+:** prima di reset, copiare sempre CSV in cartella ANALISI corrispondente.

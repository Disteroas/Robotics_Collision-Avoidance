# Esperimenti — log comparativo

Tutti i training eseguiti su questo progetto, dal più vecchio al più recente.

---

## Panoramica rapida

| # | Branch | Maze | Episodi | M2 test | M3 test | Esito |
|---|--------|------|---------|---------|---------|-------|
| 1 | `main` (baseline) | 1 | ~500 | 0% | 0% | Fallito — rete base |
| 2 | `curriculum_learning` | 1→2 | 3000 | 0% | 0% | Fallito — vedi sotto |
| 3 | `paper_implementation` | 1→2 (curriculum) | 6115 | 0% | 0% | Fallito — 5 cause identificate |
| 4 | **`feng_direct`** | 2 | 3000 | **10%** | 0% | Parziale — primo successo |
| 5 | `fixed_feng` | 2 | 3000 | N/D | N/D | Fallito — avg100 < 0, modifiche errate |
| 6 | `merge11_05` | M1+M2 | 5000 | 0% | 0% | Fallito — MAX_STEPS bug |
| 7 | `merge12_05` | M1+M2 | 4500 | **46.7%** | 0% | Prima convergenza stabile |
| 8 | `merge14_05` run3 | M2-only | 4000 | 20% | **13.3%** | **Unica generalizzazione M3** |
| 9 | `merge15_05` | M2-only | 8000 | 13% | 0% | Overfitting posizionale |
| 10 | `merge16_05` run1 | M2-only | 5000 | **46%** | 0% | Dead-end exploitation |
| 10 | `merge16_05` run2 | M2-only | 5000 | 33% | 0% | Policy degradation |

---

## Esperimento 1 — Baseline (`main`)

**Configurazione:** spawn fisso, reward semplice, nessun curriculum.  
**Risultato:** crash rate ~100% su tutti i maze. Usato solo come infrastruttura di partenza.

---

## Esperimento 2 — `curriculum_learning`

**Configurazione:** Phase 1 (Maze 1) → Phase 2 (Maze 2) al superamento della soglia reward.  
**Reward:** complessa (spazio libero, danger zone front/side, steering penalty).  
**Risultato:** 3000 ep, crash rate ~73% su Maze 2. 0 successi.

**Cause:** reward troppo densa mascherava mancanza di apprendimento reale. Phase transition errata.

---

## Esperimento 3 — `paper_implementation`

**Branch:** `paper_implementation`  
**Data:** 2026-05-07  
**Configurazione:**
- Maze 1 (Phase 1) → Maze 2 (Phase 2, curriculum)
- BETA_DECAY = 0.995 (errore: ε a 0.05 già a ep 600)
- PHASE2_THRESHOLD su avg reward, non su success rate
- Spawn fisso per episodio

**Training:** 6115 episodi totali.

**Risultati test:**

| Maze | Crash rate | Successi |
|------|-----------|---------|
| Maze 1 | >85% | 0/30 |
| Maze 2 | >85% | 0/30 |
| Maze 3 | >85% | 0/30 |

**5 cause radice identificate:**
1. `BETA_DECAY=0.995` → ε a 0.05 a ep 600 anziché 3000 (troppo presto)
2. `PHASE2_THRESHOLD` su avg reward → Phase 2 attivata con 80% crash rate
3. Catastrophic forgetting: Maze 2 degrada policy Maze 1
4. Spawn fisso → reward densa maschera mancanza apprendimento
5. Reset ε a Phase 2 ininfluente

**Analisi completa:** `risultati/PAPER_IMPLEMENTATION_SESSION.md`

---

## Esperimento 4 — `feng_direct`

**Branch:** `feng_direct`  
**Data:** 2026-05-08  
**Configurazione:**
- Maze 2 only (no curriculum)
- BETA_DECAY = 0.999 (ε → 0.05 a ep ~3000)
- 16 spawn random per-episodio, 6 zone, safety check post-teleport
- Reward: +5/-1000 (binario)

**Training:** 3000 episodi.

| Metrica training | Valore |
|-----------------|--------|
| Epsilon finale | 0.050 |
| Avg-100 finale | **+391** (da -400 iniziale) |
| Crash rate | 99.6% (2987/3000) |
| Steps totali | 535,354 |
| Primo episodio completato | ep **1996** (ε=0.136) |

**Curva apprendimento — avg-100:**

```
ep    1:  -400  (esplorazione pura)
ep  500:  -433  (ancora poco)
ep 1000:  -342  (episodi più lunghi)
ep 1500:  -156  (episodi 400+ step)
ep 1996:   +34  ← primo successo completo (1000 step)
ep 2500:  +134
ep 3000:  +391  ← fine training, curva ancora in salita
```

**Risultati test (best_ddqn_model.pth, ε=0.0):**

| Maze | Crash rate | Successi | Avg reward | Note |
|------|-----------|---------|-----------|------|
| Maze 1 (mai visto) | 100% | 0/30 | -487 | Nessuna generalizzazione |
| Maze 2 (training) | **90%** | **3/30** | -345 | 7 crash step=1 per bug spawn* |
| Maze 3 (mai visto) | 100% | 0/30 | -466 | Nessuna generalizzazione |

*Bug `test.py`: `reset_environment()` senza `maze_id` → spawn da SPAWN_LISTS[1] in Maze 2/3. **Fixato nel commit `4bbc476`.**

**Analisi completa:** [report_feng_direct.md](report_feng_direct.md)

**Giudizio:** Miglioramento strutturale rispetto a `paper_implementation` (avg100 positivo, 3 successi), ma cause architetturali bloccano la convergenza. Vedere [NEXT_STEPS.md](NEXT_STEPS.md).

---

## Esperimento 5 — `fixed_feng` ← FALLITO

**Branch:** `fixed_feng`  
**Data:** 2026-05-09  
**Autore modifiche:** BoloM03 (Matteo Bolo)  
**Configurazione:** identica a `feng_direct` + 3 modifiche in `train_core.py`:
- BATCH_SIZE: 64 → 256
- Loss: MSELoss → SmoothL1Loss (Huber δ=1)
- grad_clip: 10.0 → 1.0

**Training:** 3000 episodi.

| Metrica | Valore |
|---------|--------|
| Avg-100 finale | **< 0** (peggiorato vs feng_direct +391) |
| Loss | Bassa (bad fixed point — Q-values costanti) |

**Causa fallimento:** Huber(δ=1) + clip=1.0 riduce il segnale di apprendimento dei crash di ~10.000× rispetto a MSE + clip=10.0. La rete converge a predire Q-values costanti negativi — loss bassa ma policy inutile. Feng 2021 usa MSE pura e non menziona grad_clip.

**Errori nell'analisi di Matteo (`ANALISI_PARAMETRI_FENG.md`):**
- PER indicato come soluzione → Feng lo ha testato e scartato (reward finale peggiore)
- MSE indicata come instabile → funziona nel paper e in `feng_direct`

**Analisi completa:** [ANALISI_FIXED_FENG_FALLIMENTO.md](PAPER_ANALYSIS/ANALISI_FIXED_FENG_FALLIMENTO.md)

---

## Esperimento 6 — `merge11_05` ← COMPLETATO

**Branch:** `merge11_05`  
**Data:** 2026-05-11/12  
**Configurazione:**
- Maze 1 + Maze 2 interleaved (pattern M1/M2/M2, ratio 1:2)
- 5000 ep totali, 25 blocchi × 200 ep
- BETA_DECAY=0.999 (ε → 0.05 a ep ~3000), epsilon mai resettato
- 2 spawn Maze 1: P1 (-2.9,-2.0,N) canale sinistro, P2 (1.0,-1.0,N) camera interna
- Reward: +5/-1000 (semplice)
- GAZEBO_SPEED=5×
- **MAX_STEPS=1000 ← BUG** (training vs test mismatch)

**Risultati test (30 ep/maze, ε=0.0, spawn deterministici):**

| Maze | Success rate | Avg reward | Note |
|------|-------------|-----------|------|
| M1 | **57% (17/30)** | ~+1193 | P1 sempre OK; P2 sempre crash step ~97-98 |
| M2 | **0% (0/30)** | ~+215 | Crash step 131-357; 6 cluster comportamentali |
| M3 | **0% (0/30)** | ~-577 | Crash step 85-88; nessuna generalizzazione |

**Baseline valida (randomSpawn 05_08):** M1=30%, M2=33%, M3=33%

**Analisi risultati:**
- M1: 57% > 30% ✓ multi-maze migliora M1 rispetto a training single-maze
- M2: 0% << 33% ✗ **causa root: MAX_STEPS=1000** in training. Con 1000 step, M2 non è quasi mai completabile → nessun reward positivo → policy M2 non impara
- M3: 0% ✗ dipende da M2: se M2 non convergito, la policy non può generalizzare

**M1 analisi spawn:**
- P1 (-2.9,-2.0): 17/17 successi (100%) — canale sinistro, traiettoria pulita
- P2 (1.0,-1.0): 0/13 successi (0%), sempre crash step ~97-98 — possibile trappola geometrica

**M2 analisi cluster (steps):**
- Cluster "quasi completo": step 348-357, reward ~+780 (70% episodio, poi crash a chokepoint)
- Cluster "medio": step 252-255, reward ~+260
- Cluster "corto": step 131-143, reward -290/-350

**Causa M2 training failure:** randomSpawn 05_08 (MAX_STEPS=500) → 52% M2 training success. merge11_05 (MAX_STEPS=1000) → ~0% M2 training success. Il MAX_STEPS mismatch è la causa principale.

**Spec:** `docs/superpowers/specs/2026-05-11-multimaze-training-design.md`

---

## Esperimento 7 — `merge12_05` ← COMPLETATO

**Branch:** `merge12_05` (da `merge11_05`)  
**Data:** 2026-05-12/14  
**Configurazione:**
- Maze 1 + Maze 2 interleaved (pattern M1/M2/M2, ratio 1:2)
- 4500 ep totali, 45 blocchi × 100 ep
- BETA_DECAY=0.999, MAX_STEPS=500, best_avg in checkpoint
- 2 spawn Maze 1: P1, P2. 10 spawn Maze 2 (rimossi A2, B1, B2, C1, C3, E1)

**Risultati test (30 ep/maze, ε=0.0, spawn random da TEST_SPAWN_LISTS):**

| Maze | Success rate | Avg reward | vs Baseline | vs merge11_05 |
|------|-------------|-----------|-------------|---------------|
| M1 | **20/30 (66.7%)** | 2500 (succ) / -570 (crash) | +40pp ✓ | +10pp ✓ |
| M2 | **14/30 (46.7%)** | 2500 (succ) | +20pp ✓ | +47pp ✓ |
| M3 | **0/30 (0%)** | ~+180 | -40pp ✗ | 0pp |

Baseline: randomSpawn 05_08 — M1=26.7%, M2=26.7%, M3=40%.

**Cluster crash M1 (10 ep):** step 86-88, reward -570 → P2 (1.0,-1.0,N) sempre fallisce. P1 100% successi.

**Cluster crash M2 (16 ep):**
- Tipo A (5 ep): step=345, reward=+720 — crash a chokepoint (quasi completo)
- Tipo B (9 ep): step=113, reward=-440 — crash precoce (spawn più difficile, prob. B3)
- Tipo C/D (2 ep): step ~290, reward ~+445

**M3:** 0% — nessuna generalizzazione. Multi-maze M1+M2 peggiora vs M2-only (40%→0%). Causa: policy spostata verso M1, riduce similarità M2→M3.

**Baseline corretta (da CSV):** M1=26.7% (8/30), M2=26.7% (8/30), M3=40% (12/30).  
Nota: valori "33%" in documenti precedenti erano approssimazioni.

**Spec:** `docs/superpowers/specs/2026-05-12-merge12-training-design.md`

---

## Esperimento 8 — `merge14_05` ← COMPLETATO

**Branch:** `merge14_05` (da `merge12_05`)  
**Data implementazione:** 2026-05-14  
**Configurazione:**
- Maze 2 only (`BLOCK_PATTERN=(2)`)
- **4000 ep totali, 20 blocchi × 200 ep**
- BETA_DECAY=0.999 (invariato), MAX_STEPS=500 (invariato)
- **REPLAY_START_SIZE=10,000** (fix bug prefill — era BATCH_SIZE=64)
- Reward: +5/-1000 (binaria)
- 10 spawn Maze 2 training (invariati), 6 spawn test (invariati)
- TARGET_UPDATE=1000

**Risultati test run3 (best model, 30 ep/maze, ε=0.0):**

| Maze | Success rate | Avg steps (crash) | Fenomeno |
|------|-------------|------------------|----------|
| M2 (training) | **20%** | — | Policy grezza ma funzionale |
| M3 (unseen) | **13.3%** | 345 (crash) | **Unica generalizzazione M3 osservata** |

**Nota:** Run3 = vecchio seed. Run con nuovo seed (seed 123) → 0% su tutti i maze (seed brittleness).

**Finding chiave:** Unica run dell'intero progetto con M3 > 0%. Mirowski et al. 2016: ~345 step prima del crash su M3 indica navigazione attiva, non crash immediato. Policy ha appreso relazione "scan basso → virata" in modo parzialmente generalizzabile.

**Causa M3=13%:** Convergenza parziale, non overfitting. Policy non ancora specializzata su traiettorie M2-specifiche → mantiene generalizzabilità parziale. Confermato: merge15_05 (più episodi su M2) → M3=0% per overfitting.

**Spec:** `docs/superpowers/specs/2026-05-14-merge14-training-design.md`  
**Piano:** `docs/superpowers/plans/2026-05-14-merge14-training.md`

---

## Esperimento 9 — `merge15_05` ← COMPLETATO

**Branch:** `merge15_05` (da `merge14_05`)  
**Data:** 2026-05-15  
**Configurazione:**
- Maze 2 only, **8000 ep totali**
- TARGET_UPDATE=1000 (invariato)
- Reward: +5/-1000 (binaria)
- 6 spawn training M2, spawn point M3: (-2.5, -0.25, 0.0) [fix da (-2.0,-1.0,0.0)]
- set_seed(42) aggiunto da Matteo (BoloM03)

**Risultati training:**

| Metrica | Valore |
|---------|--------|
| Best avg100 | **+1365** @ ep ~6000 |
| Final avg100 | ~+300 (instabile) |
| Crash rate finale | ~80% |
| Oscillazione avg100 | ±800 pts (instabilità cronica) |

**Risultati test (90 ep, ε=0.0):**

| Maze | Success rate | Fenomeno |
|------|-------------|----------|
| M2 (training) | **13%** | Peggiorato vs merge14_05 (20%)! |
| M3 (unseen) | **0%** | Zero generalizzazione |

**Findings:**
1. **Overfitting posizionale:** +8000 ep su M2 → policy memorizza traiettorie specifiche. Reward avg100 alta in training non implica buona policy di test.
2. **Oscillazione cronica ±800 pts:** TARGET_UPDATE=1000 troppo frequente (Van Hasselt 2016). Con ~13 gradient steps per target shift → Q-values oscillano → policy instabile.
3. **set_seed(42) multi-blocco:** reset RNG a seed 42 ogni blocco → esplorazione bias sistematico per ogni blocco (non genuinamente random).
4. **M3=0%:** Training prolungato su singolo maze → specializzazione → perdita generalizzazione.

**Causa M2=13% < merge14_05 (20%):** Paradosso più episodi → peggiore performance. Overfitting traiettorie: ε=0.05 troppo basso per rompere loop deterministici. La policy ha memorizzato 1-2 percorsi, non apprende navigazione generale.

---

## Esperimento 10 — `merge16_05` ← COMPLETATO (2 run)

**Branch:** `merge16_05` (da `merge15_05`)  
**Data:** 2026-05-16/17  
**Configurazione:**
- Maze 2 only, **5000 ep totali**
- **TARGET_UPDATE=5000** (fix da 1000 — Van Hasselt 2016)
- **B3 rimosso** da SPAWN_LISTS[2] (6 → 5 spawn)
- **Reward shaping:** +5/step +space_bonus −front_danger −side_danger −steering_penalty −1000 collision
  - `space_bonus = 2.0 × mean(ALL 50 beams) / 5.0`
  - `front_danger = 1.5m` threshold, `side_danger = 0.45m` threshold
- **TEST_SPAWN_LISTS[2] = SPAWN_LISTS[2]** (fix allineamento)
- LIDAR: uniform sampling (Matteo, non min-pooling — regressione non corretta in questo branch)

**Risultati training:**

| Metrica | Run 1 | Run 2 |
|---------|-------|-------|
| Final avg100 | **+197.5** | **-65.4** (policy degradation!) |
| Best avg100 | +354.3 @ ep4945 | +360.2 @ ep3601 |
| Crash rate totale | 80.8% | 81.1% |
| avg100 > 0 da | ep 2644 | ep 2652 |

**Spawn breakdown training (run1 / run2):**

| Spawn | Run1 max-steps% | Run2 max-steps% | Verdetto |
|-------|----------------|----------------|---------|
| F1 (-4.5,-3.5) | 60.3% | 63.8% | OK |
| F3 (6.0,6.0) | 29.5% | 47.5% | OK |
| A1 (-6.0,0.0) | 14.0% | 1.6% | OK (seed brittleness) |
| D1 (1.5,0.0) | 11.9% | **0%** | Marginale |
| **F2 (-1.5,-4.0)** | **0%** | **0%** | **DEAD-END STRUTTURALE** |
| **C2 (-7.0,5.0)** | **0%** | **0.1%** | **DEAD-END STRUTTURALE** |

**Risultati test M2 (90 ep, ε=0.0):**

| Spawn | Run1 success | Run2 success |
|-------|-------------|-------------|
| F1 (-4.5,-3.5) | **100%** | **100%** |
| A1 (-6.0,0.0) | **100%** | 0% ← seed brittleness |
| F3 (6.0,6.0) | 0% ← seed brittleness | **100%** |
| D1 (1.5,0.0) | 15% | 0% |
| F2 (-1.5,-4.0) | **0%** | **0%** |
| C2 (-7.0,5.0) | **0%** | **0%** |
| **Global** | **46%** | **33%** |

**M3:** 0% entrambe le run.

**5 Findings:**
1. **Dead-end exploitation (F2, C2):** `space_bonus = 2.0 × mean(ALL beams)` premia open space indipendentemente dalla direzione. Dead-end con fronte aperta → space_bonus alto → policy preferisce oscillare nel dead-end. Non-potential shaping (Ng et al. 1999) → local optimum spurio (Devlin & Kudenko 2012). **Reward hacking** confermato strutturale su 2 run.
2. **Seed brittleness estrema:** A1=100%/F3=0% in run1 vs A1=0%/F3=100% in run2. Trajectory memorization seed-dipendente (Henderson et al. 2018).
3. **Policy degradation run2:** 360→-65 in 1400 ep. TARGET_UPDATE=5000 ha ridotto oscillazione (±800→±250 pts) ma non ha eliminato degrado a lungo termine.
4. **POMDP aliasing (causa primaria):** Senza heading nello stato, stesso vettore LIDAR in corridoio vs dead-end → stessa azione (Mirowski et al. 2016).
5. **Assenza goal:** space_bonus sostituisce gradiente direzionale in modo non-potential → local optima (Tai et al. 2017, Zhu et al. 2017).

**Spawn da rimuovere in merge17_05:** F2 (-1.5,-4.0) e C2 (-7.0,5.0) → 4 spawn rimanenti: F1, F3, A1, D1.

**Analisi completa:** `risultati/merge16_05_analysis.md`

---

## Plot

Tutti i plot di `feng_direct` sono in `analysis/plots/feng_direct/`:
- `00_dashboard.png` — overview completo
- `01_training_reward.png` — reward per episodio + avg-100
- `02_training_crash_rate.png` — crash rate rolling 100 ep
- `03_training_epsilon_loss.png` — epsilon + loss
- `04_test_results.png` — crash rate e distribuzione reward test

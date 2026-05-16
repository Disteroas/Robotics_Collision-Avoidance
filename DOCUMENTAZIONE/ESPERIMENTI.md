# Esperimenti — log comparativo

Tutti i training eseguiti su questo progetto, dal più vecchio al più recente.

---

## Panoramica rapida

| # | Branch | Maze | Episodi | Crash rate test M2 | Successi M2 | Esito |
|---|--------|------|---------|---------------------|-------------|-------|
| 1 | `main` (baseline) | 1 | ~500 | ~100% | 0 | Fallito — rete base |
| 2 | `curriculum_learning` | 1→2 | 3000 | ~73% | 0/30 | Fallito — vedi sotto |
| 3 | `paper_implementation` | 1→2 (curriculum) | 6115 | >85% tutti | 0/30 | Fallito — 5 cause identificate |
| 4 | **`feng_direct`** | 2 | 3000 | **90%** | **3/30** | Parziale — primo successo |
| 5 | `fixed_feng` | 2 | 3000 | N/D | N/D | Fallito — avg100 < 0, modifiche errate |
| 6 | `merge11_05` | M1+M2 interleaved | 5000 (pianificato) | — | — | **IN PROGRESS** — training non ancora avviato |
| 7 | `merge12_05` | M2 only | 4500 | 53.3% | 14/30 | **Parziale** — M1=66.7%, M2=46.7%, M3=0% |
| 8 | `merge14_05` | M2 only | 4000 | — | — | **Parziale** — run3: M2=20%, M3=13% zero-shot |
| 9 | `merge15_05` | M2 only | 8000 | 87% | 12/90 (13%) | **Parziale** — solo F1 funziona, overfitting posizionale |
| 10 | `merge16_05` | M2 only | 5000 | — | — | **IN PROGRESS** — reward shaping + target net fix |

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

## Esperimento 8 — `merge14_05` ← IMPLEMENTATO (training da avviare)

**Branch:** `merge14_05` (da `merge12_05`)  
**Data implementazione:** 2026-05-14  
**Configurazione:**
- Maze 2 only (`BLOCK_PATTERN=(2)`)
- **4000 ep totali, 20 blocchi × 200 ep**
- BETA_DECAY=0.999 (invariato), MAX_STEPS=500 (invariato)
- **REPLAY_START_SIZE=10,000** (fix bug prefill — era BATCH_SIZE=64)
- 10 spawn Maze 2 training (invariati), 6 spawn test (invariati)
- Spawn point loggato nel CSV training (`spawn` colonna) e nel terminale
- best_avg in checkpoint (invariato)
- Prefill completato ~ep 155 (10k step / ~65 step/ep)

**Motivazione fix:**
1. **Prefill (Mnih 2015):** buffer iniziale da 10k transizioni random → diversità iid garantita → gradiente stabile. Loss curve merge12_05 (110→2919→5849) eliminata.
2. **M2-only (negative transfer):** M1+M2 training azzerava M3=0%. M2→M3 similarità geometrica ripristina generalizzazione (baseline: M3=40% con M2-only).
3. **Spawn logging:** colonna `spawn` in training_log.csv per identificare cluster crash per spawn point.

**Avvio training:** `./start_train_multimaze.sh --reset`

**Target:**

| Metrica | Target | Baseline (randomSpawn 05_08) | merge12_05 |
|---------|--------|------------------------------|-----------|
| M1 | — | 26.7% | 66.7% |
| M2 | ≥ 50% | 26.7% | 46.7% |
| M3 | ≥ 40% | **40.0%** | 0% |

**Spec:** `docs/superpowers/specs/2026-05-14-merge14-training-design.md`  
**Piano:** `docs/superpowers/plans/2026-05-14-merge14-training.md`

---

## Esperimento 9 — `merge15_05` ← COMPLETATO

**Branch:** `merge15_05` (da `merge14_05`)
**Data implementazione:** 2026-05-15  
**Data training/test:** 2026-05-16

**Configurazione:**
- Maze 2 only (`BLOCK_PATTERN=(2)`)
- **8000 ep totali, 40 blocchi × 200 ep**
- BETA_DECAY=0.999 (invariato), MAX_STEPS=500 (invariato)
- **REPLAY_START_SIZE=10,000** (invariato), TARGET_UPDATE_STEPS=1,000
- **7 spawn M2 training** (rimossi D2 (0.5,-2.0), D3 (3.5,0.5), E2 (0.0,3.5))
- Reward: +5 / -1000 (binario puro, invariato)
- Ripartenza da zero (checkpoint non riutilizzato)

**Motivazione:**
1. **Non-convergenza merge14_05:** curva avg100 ancora in salita a ep 4000 in run3 (peak@ep3818, final avg100=700). 8000 ep per raggiungere convergenza.
2. **Spawn tossici rimossi:** D2/D3/E2 = 0% completion su ~1180 ep totali (3 run). Avg steps 55-128. Pure gradient noise nel replay buffer.
3. **Ripartenza pulita:** validità scientifica — mix di regime vecchio/nuovo invalida l'analisi comparativa.
4. **Test a 90 ep/maze:** riduce CI da ±18pp a ±10pp per confronti affidabili.

**Risultati:**

| Metrica | Risultato | Target | Note |
|---------|-----------|--------|------|
| Best avg100 | 1366 @ ep 7066 | — | Record storico |
| Final avg100 | 845.9 | — | Regressione da best (-520 pts) |
| Crash rate last 100 | 82% | — | Non converge |
| M1 test | 0% | — | Atteso (M2-only) |
| M2 test | **13%** | ≥ 50% | ❌ Solo F1 (100%), tutti altri 0% |
| M3 test | **0%** | ≥ 40% | ❌ Avg 75 step, crash immediato |

**Spawn training (ordinati per max-steps rate):**

| Spawn | Avg steps | Max-steps rate | Classificazione |
|---|---|---|---|
| F1 (-4.5,-3.5) | 366.5 | 51.0% | Apprendimento principale ✅ |
| A1 (-6.0,0.0) | 280.1 | 26.2% | Buon segnale ✅ |
| D1 (1.5,0.0) | 156.6 | 12.2% | Bimodale: 142 completamenti reali ✅ |
| C2 (-7.0,5.0) | 328.4 | 10.5% | Segnale medio |
| F2 (-1.5,-4.0) | 430.4 | 5.1% | Sopravvive lungo, non completa |
| F3 (6.0,6.0) | 320.9 | 2.8% | Borderline |
| B3 (-4.5,1.5) | 242.2 | **0.0%** | **TOSSICO** — 1137 ep, 0 completamenti |

**Findings chiave:**
1. **Training instabile:** reward curve oscilla ±800 pts senza convergenza. Causa: TARGET_UPDATE=1000 (15 gradient steps per target shift — insufficiente).
2. **Overfitting posizionale:** M2 13% = F1 100% + tutti altri 0%. Memorizzazione traiettoria, non obstacle avoidance generale.
3. **Gap training→test:** A1 26% training → 0% test. Il 26% include episodi salvati da perturbazioni random (ε=0.05), non riproducibili in greedy test. POMDP aliasing confermato.
4. **B3 tossico confermato:** 0% su 8000 ep (1137 ep) = gradient noise strutturale.
5. **avg100 fuorviante:** crash a 430 step dà ~+720 reward pur fallendo. Best avg100=1366 ≠ buona policy.
6. **Seed singola:** varianza ignota. merge14_05 su 3 run: 0%/30%/20% → merge15_05 13% potrebbe essere seed sfavorevole.

**Analisi completa:** `DOCUMENTAZIONE/risultati/merge15_05_analysis.md`  
**Spec:** `docs/superpowers/specs/2026-05-15-merge15-training-design.md`  
**Piano:** `docs/superpowers/plans/2026-05-15-merge15-training.md`  
**Dati:** `ANALISI_15_05/`

---

## Esperimento 10 — `merge16_05` ← IN CORSO (training da avviare)

**Branch:** `merge16_05` (da `merge15_05`)  
**Data design:** 2026-05-16

**Configurazione:**
- Maze 2 only (`BLOCK_PATTERN=(2)`)
- **5000 ep totali, 25 blocchi × 200 ep**
- BETA_DECAY=0.999 (invariato), MAX_STEPS=500 (invariato)
- REPLAY_START_SIZE=10,000 (invariato)
- **TARGET_UPDATE_STEPS: 1,000 → 5,000**
- **6 spawn M2 training** (rimosso B3 (-4.5,1.5), mantenuto D1)
- **TEST_SPAWN_LISTS[2] = SPAWN_LISTS[2]** (allineato a training — vedi motivazione punto 5)
- **Reward shaping** (da curriculum_learning, parametri tuned)
- Ripartenza da zero (fresh start)

**Reward shaping (usv_logic.py):**
```python
FRONT_DANGER = 1.5    # 30 step preavviso (v=0.5m/s, dt=0.1s)
SIDE_DANGER  = 0.45   # invariato
SPACE_BONUS_WEIGHT = 2.0

# per step:
# +5 base
# +2.0 × mean(scan)/5.0  (space bonus: 0→2.0)
# -0.02 × |action-5|     (steering: 0→0.10, soft)
# -20 × severity²        (front danger quadratic: 0→20)
# -5 × severity²  ×2     (side danger quadratic: 0→5 per lato)
```

**Motivazione:**

1. **Reward shaping** — reward binario +5/-1000 non dà gradiente pre-crash. Con shaping, approaching a 1m dal muro: penalty≈3.2/step (leggero). A 0.5m: penalty≈12.8/step (forte). Gradiente continuo: Ng et al. 1999. space_bonus penalizza wall-following loop (mean(scan) bassa vicino a pareti).

2. **TARGET_UPDATE 1000→5000** — 15 gradient steps per target shift era insufficiente (oscillazione ±800 pts in merge15_05). Con 5000: 78 gradient steps → convergenza locale prima del prossimo shift. Van Hasselt et al. 2016.

3. **B3 rimosso** — 0% max-steps su 8000 ep (1137 ep) = gradient noise strutturale. D1 mantenuto: 12.2% max-steps = 142 completamenti reali (distribuzione bimodale, non tossico).

4. **5000 ep** — Reward denso → sample efficiency stimata +30-40% (Grzes & Kudenko 2009). Best model salvato automaticamente: safe stop anticipato se converge prima.

5. **TEST_SPAWN_LISTS[2] = SPAWN_LISTS[2]** — In merge15_05, i punti test M2 includevano B3/D2/E2 (rimossi dal training): davano 0% per ragioni geometriche, non per debolezza della policy. Allineare test a training risolve l'ambiguità: la metrica M2 misura "quali spawn ha risolto la policy?" invece di generalization intra-maze. La generalization cross-environment è già coperta da M3 (zero-shot, mai vista in training). Cobbe et al. 2019 ("Quantifying Generalization in Reinforcement Learning"): un test set misura generalization *solo se diverso* dal training set; se si vuole misurare la qualità della policy su spawn conosciuti, il test set deve coincidere con quelli di training.

**Avvio training:** `./start_train_multimaze.sh --reset`

**Target:**

| Metrica | Target realistico | Target ottimistico | Baseline merge15_05 |
|---------|------------------|-------------------|---------------------|
| M2 test | ≥ 30% | ≥ 50% | 13% |
| M3 test | ≥ 5% | ≥ 20% | 0% |

**Nota:** target M2≥50% era ottimistico. POMDP aliasing (no heading) limita strutturalmente test greedy: A1/C2/F2/F3 probabilmente ancora <50% senza heading nello stato. Reward shaping migliora training stability e M2, ma non risolve il wall-following deterministico.

**Limitazioni note:**
- Se M2 < 30%: potrebbe essere seed sfavorevole o reward shaping controproducente per geometria stretta M2
- Se training stabile (no oscillazione) ma M2 < 30%: bottleneck è POMDP aliasing → merge17_05 con heading
- Completion bonus (+200 a MAX_STEPS): analizzato e scartato. γ^500 × 200 ≈ 1.3 punti da step 0 — irrilevante vs step reward. Non implementato.

---

## Plot

Tutti i plot di `feng_direct` sono in `analysis/plots/feng_direct/`:
- `00_dashboard.png` — overview completo
- `01_training_reward.png` — reward per episodio + avg-100
- `02_training_crash_rate.png` — crash rate rolling 100 ep
- `03_training_epsilon_loss.png` — epsilon + loss
- `04_test_results.png` — crash rate e distribuzione reward test

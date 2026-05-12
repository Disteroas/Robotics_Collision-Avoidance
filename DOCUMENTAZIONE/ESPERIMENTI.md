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

**Analisi completa:** [ANALISI_FIXED_FENG_FALLIMENTO.md](ANALISI_FIXED_FENG_FALLIMENTO.md)

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

## Esperimento 7 — `merge12_05` ← PIANIFICATO

**Branch:** `merge12_05` (da `merge11_05`)  
**Data pianificazione:** 2026-05-12  
**Configurazione:**
- Maze 1 + Maze 2 interleaved (pattern M1/M2/M2, ratio 1:2)
- **4500 ep totali, 45 blocchi × 100 ep** (was 25×200)
- BETA_DECAY=0.999 (invariato)
- 2 spawn Maze 1: P1, P2 (invariati — P2 mantenuto per diagnosi)
- Reward: +5/-1000 (invariata)
- **MAX_STEPS=500** (fix critico — era 1000)
- **best_avg persistito in checkpoint** (fix bug)
- GAZEBO_SPEED=5×

**Fix rispetto a merge11_05:**
1. MAX_STEPS: 1000 → 500 (`train.py`)
2. Blocchi: 200→100 ep, totale 25→45 (`start_train_multimaze.sh`)
3. best_avg in checkpoint (`train_core.py`)

**Target:**

| Metrica | Target | Baseline migliore |
|---------|--------|-------------------|
| Test M1 | ≥ 90% | 57% (merge11_05) |
| Test M2 | ≥ 50% | 33% (randomSpawn 05_08) |
| Test M3 | ≥ 30% | 33% (randomSpawn 05_08) |

**Spec:** `docs/superpowers/specs/2026-05-12-merge12-training-design.md`

---

## Plot

Tutti i plot di `feng_direct` sono in `analysis/plots/feng_direct/`:
- `00_dashboard.png` — overview completo
- `01_training_reward.png` — reward per episodio + avg-100
- `02_training_crash_rate.png` — crash rate rolling 100 ep
- `03_training_epsilon_loss.png` — epsilon + loss
- `04_test_results.png` — crash rate e distribuzione reward test

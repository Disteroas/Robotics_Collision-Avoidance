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

## Plot

Tutti i plot di `feng_direct` sono in `analysis/plots/feng_direct/`:
- `00_dashboard.png` — overview completo
- `01_training_reward.png` — reward per episodio + avg-100
- `02_training_crash_rate.png` — crash rate rolling 100 ep
- `03_training_epsilon_loss.png` — epsilon + loss
- `04_test_results.png` — crash rate e distribuzione reward test

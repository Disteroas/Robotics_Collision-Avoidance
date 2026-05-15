# Analisi merge14_05 — 3 run indipendenti

**Data analisi:** 2026-05-15  
**Branch:** merge14_05  
**Analisi automatica:** `analysis/analysis_v_15_05.py`

---

## Configurazione

| Parametro | Valore |
|---|---|
| Episodi totali | 4000 |
| Blocchi | 20 × 200 ep |
| Pattern | M2-only (BLOCK_PATTERN=(2)) |
| MAX_STEPS | 500 |
| BETA_DECAY | 0.999 (ε: 1.0→0.05 in ~3000 ep) |
| REPLAY_START_SIZE | 10,000 |
| Spawn training M2 | 10 punti (SPAWN_LISTS[2]) |
| Spawn test M2 | 6 punti (TEST_SPAWN_LISTS[2]) |

---

## Risultati per run

| Run | Macchina | Peak avg100 | @ep | Final avg100 | M2 test | M3 test (zero-shot) |
|-----|----------|------------|-----|-------------|---------|---------------------|
| run1 | PC principale | ~750 | 3115 | — | 0% | — |
| run2 | PC principale | ~830 | 3587 | — | 30% | — |
| run3 | PC Matteo | **920.3** | **3818** | **700.0** | 20% | **13%** |

**Crash rate (last 100 ep, run3):** 81.0%  
**Crash rate (all ep, run3):** 92.6%

---

## Analisi spawn (run3 — più completa)

Dati da `ANALISI_14_05/run3/summary_training.txt`:

| Spawn | Label | Avg steps | Max-steps rate | Classificazione |
|-------|-------|-----------|----------------|-----------------|
| (-1.5,-4.0) | F2 | 385.3 | 0.3% | Segnale ricco — max avg steps, ~75% maze percorso prima del bottleneck. **TENERE** |
| (6.0,6.0) | F3 | 343.1 | 25.0% | Spawn di apprendimento secondario. **TENERE** |
| (-4.5,-3.5) | F1 | 335.2 | 37.8% | Spawn di apprendimento principale. **TENERE** |
| (-7.0,5.0) | C2 | 300.1 | 0.0% | Segnale medio, consistente. **TENERE** |
| (-4.5,1.5) | B3 | 220.8 | 0.0% | Segnale medio, diversità maze. **TENERE** |
| (-6.0,0.0) | A1 | 220.2 | 8.4% | In sblocco (1.5%→6.9%→8.4% su 3 run). **TENERE** |
| (0.0,3.5) | E2 | 133.1 | 0.0% | Crash breve, 0% su 3 run. **RIMOSSO in merge15_05** |
| (0.5,-2.0) | D2 | 112.0 | 0.0% | Crash breve, 0% su 3 run. **RIMOSSO in merge15_05** |
| (1.5,0.0) | D1 | 111.6 | 5.1% | Borderline — alcuni completamenti. **TENERE** |
| (3.5,0.5) | D3 | 54.6 | 0.0% | Near-instant crash, 0% su 3 run. **RIMOSSO in merge15_05** |

---

## Findings chiave

### 1. Non-convergenza a 4000 episodi

Il trend del picco avg100 attraverso le 3 run mostra che il training non è completo:

| Run | Peak avg100 | @ep |
|-----|------------|-----|
| run1 | ~750 | 3115 |
| run2 | ~830 | 3587 |
| run3 | 920 | **3818** — curva ancora in salita all'ultimo blocco |

Il picco si sposta verso la fine del training ad ogni run, indicando che 4000 ep non bastano. **Soluzione merge15_05: 8000 ep.**

### 2. Spawn tossici strutturalmente determinati

La classificazione degli spawn è identica su 3 run su 2 macchine diverse — non è varianza stocastica:

- **D3** (3.5, 0.5): avg 54-55 step su tutte le run → crash quasi istantaneo ad ogni episodio. Probabile posizione di partenza faccia-al-muro.
- **D2** (0.5, -2.0): avg 110-112 step, 0% completion. Corridoio senza uscita dalla posizione iniziale.
- **E2** (0.0, 3.5): avg 128-133 step, 0% completion. Pattern identico su 3 run.

Questi 3 spawn generano ~1180 episodi totali (3 run × ~395 ep each) di pure crash transitions nel replay buffer — gradient noise senza segnale utile.

### 3. Generalizzazione M3 zero-shot

Run3 (avg100=700, ancora in salita) ha ottenuto **M3=13% senza alcun training su M3**. Questo è il primo segnale positivo di generalizzazione nel progetto.

**Ipotesi:** la generalizzazione emerge quando la policy M2 supera una soglia di qualità (avg100 ≈ 700+ stabile). Più training → qualità più alta → maggiore generalizzazione zero-shot.

### 4. Alta varianza test con 30 episodi

Test identici su merge14_05 hanno prodotto 0%/30%/20% su M2 (3 run diverse). Con 30 ep (5 per spawn × 6 spawn), l'intervallo di confidenza è ±18pp — insufficiente per confronti affidabili.

**Soluzione merge15_05:** EPISODES_PER_MAZE=90 (15 per spawn), CI ±10pp.

---

## Decisioni per merge15_05

| Decisione | Motivazione |
|---|---|
| 8000 ep (TOTAL_BLOCKS=40) | Curva ancora in salita a ep 3818 in run3 |
| Rimuovi D2, D3, E2 da SPAWN_LISTS[2] | 0% completion su 3 run indipendenti, avg <130 step |
| Fresh start (--reset) | Validità scientifica: no mix regime 10-spawn/7-spawn nel buffer |
| EPISODES_PER_MAZE=90 | CI ±10pp vs ±18pp con 30 ep |
| TEST_SPAWN_LISTS[2] invariato | D2 e E2 rimangono per riproducibilità test comparativi |
| M2-only (no M3 training) | M3 è il set di generalizzazione — trainare su M3 invalida la metrica |

---

## File di analisi

- Plot automatici: `analysis/analysis_v_15_05.py`
  - `01_training_curve.png` — avg100 + reward raw
  - `02_spawn_breakdown.png` — per-spawn heatmap
  - `03_crash_rate.png` — crash rate rolling
  - `04_loss_curve.png` — loss MSE (linear + log scale) ← aggiunto in merge15_05
  - `05_test_M1.png`, `06_test_M2.png`, `07_test_M3.png` — test per maze
- Raw data: `ANALISI_14_05/run{1,2,3}/training_log.csv`
- Summaries: `ANALISI_14_05/run{1,2,3}/summary_training.txt`

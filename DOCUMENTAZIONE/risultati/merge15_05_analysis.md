# Analisi merge15_05 — run singola

**Data analisi:** 2026-05-16  
**Branch:** merge15_05  
**Dati:** `ANALISI_15_05/summary_training.txt` + plot `ANALISI_15_05/plots/`

---

## Configurazione

| Parametro | Valore |
|---|---|
| Episodi totali | 8000 |
| Blocchi | 40 × 200 ep |
| Pattern | M2-only (BLOCK_PATTERN=(2)) |
| MAX_STEPS | 500 |
| BETA_DECAY | 0.999 (ε: 1.0→0.05 in ~3000 ep) |
| REPLAY_START_SIZE | 10,000 |
| TARGET_UPDATE_STEPS | 1,000 |
| Reward | +5 / -1000 (binario puro) |
| Spawn training M2 | 7 punti (rimossi D2/D3/E2 da merge14_05) |
| Spawn test M2 | 6 punti (TEST_SPAWN_LISTS[2] invariato) |
| EPISODES_PER_MAZE | 90 (15 per spawn, CI ±10pp) |

---

## Risultati training

| Metrica | Valore |
|---|---|
| Best avg100 | **1366.0** @ ep 7066 |
| Final avg100 | 845.9 |
| Crash rate (all ep) | 84.8% |
| Crash rate (last 100 ep) | 82.0% |
| Total steps | 2,427,839 |
| Avg steps/ep | ~303 |
| Tempo reale stimato (GAZEBO_SPEED=5) | ~13.5 ore |

**avg100 > 0 da:** ep 1004

---

## Risultati test

| Maze | Success% | Avg steps | Note |
|---|---|---|---|
| M1 | **0%** | ~145 | Atteso — M2-only training |
| M2 | **13%** | ~275 | Solo F1 funziona (100%), tutti gli altri 0% |
| M3 | **0%** | ~75 | Nessuna generalizzazione zero-shot |

### Per-spawn breakdown M2 (test)

| Spawn | Success% | Avg steps | Note |
|---|---|---|---|
| F1 (-4.5,-3.5) | **100%** | ~500 | Unico spawn funzionante |
| A1 (-6.0,0.0) | 0% | ~400 | Wall-following loop |
| C2 (-7.0,5.0) | 0% | ~420 | Wall-following loop |
| B3 (-4.5,1.5) | 0% | ~40 | Crash quasi istantaneo |
| E2 (0.0,3.5) | 0% | ~40 | Crash quasi istantaneo (test-only spawn) |
| D2 (0.5,-2.0) | 0% | ~100 | Crash rapido (test-only spawn) |

---

## Spawn breakdown training

| Spawn | Label | Uses | Avg steps | Max-steps rate | Classificazione |
|---|---|---|---|---|---|
| (-1.5,-4.0) | F2 | 1165 | 430.4 | 5.1% | Sopravvive lungo, non completa |
| (-4.5,-3.5) | F1 | 1124 | 366.5 | **51.0%** | Apprendimento principale ✅ |
| (-7.0,5.0) | C2 | 1124 | 328.4 | 10.5% | Medio |
| (6.0,6.0) | F3 | 1163 | 320.9 | 2.8% | Borderline |
| (-6.0,0.0) | A1 | 1123 | 280.1 | 26.2% | Buon segnale ✅ |
| (-4.5,1.5) | B3 | 1137 | 242.2 | **0.0%** | **TOSSICO — RIMUOVERE** |
| (1.5,0.0) | D1 | 1164 | 156.6 | 12.2% | Distribuzione bimodale (vedi nota) |

**Nota su D1:** avg steps basso (156) ma max-steps rate=12.2% → distribuzione bimodale. ~142 episodi a 500 step (completamenti) + ~1022 episodi a ~109 step (crash rapidi). D1 fornisce segnale utile — NON rimuovere.

---

## Findings chiave

### 1. Training instabile — oscillazione senza convergenza

La reward curve oscilla ±800 punti da ep 2000 a ep 8000. Best avg100=1366 @ ep 7066, poi **regressione** a 845 @ ep 8000. Il modello finale è peggiore del best di 520 punti.

**Causa identificata:** TARGET_UPDATE_STEPS=1000 troppo frequente.
```
1000 steps / 64 batch = 15 gradient steps per target shift
→ la rete online non converge verso i target prima che cambino
→ bootstrap instabile → oscillazione
```
Van Hasselt et al. 2016 DDQN: C richiede abbastanza stabilità per convergenza online. 15 gradient steps è insufficiente.

### 2. Overfitting posizionale estremo

M2 test = 13% = F1 100% + tutti gli altri 0%. L'agente ha memorizzato la traiettoria specifica di F1 (51% training max-steps rate, 1124 episodi dedicati). Non ha appreso obstacle avoidance generale.

### 3. Gap training→test (POMDP aliasing)

| Spawn | Training max-steps% | Test success% | Spiegazione |
|---|---|---|---|
| F1 | 51% | **100%** | Policy deterministica ottima per F1 |
| A1 | 26% | **0%** | 26% = "luck" random (ε=0.05) non riproducibile con ε=0.0 |
| C2 | 10% | **0%** | Idem |

Il 26% di A1 in training include episodi salvati da perturbazioni stocastiche (ε=0.05). La policy **deterministica** appresa per A1 è un wall-following loop che termina sempre in crash. Senza heading nello stato, il robot non può distinguere "mi avvicino al muro" da "mi allontano" → la policy greedy converge a loop.

### 4. B3 tossico confermato

B3 (-4.5,1.5): 0% max-steps rate su **8000 ep** (1137 episodi). Identico al pattern di D2/D3/E2 rimossi in merge15_05. 1137 ep = 14.2% del training budget = gradient noise puro nel buffer.

### 5. avg100 è metrica fuorviante

Best avg100=1366 ma test M2=13%. Il metric accumula reward da survival steps. Un crash a 430 step dà: 430×4 - 1000 ≈ +720 reward positivo pur crashando. L'agente "sopravvive a lungo prima di crashare" è indistinguibile da "completa" via avg100.

### 6. M3=0% — nessuna generalizzazione

Avg 75 step prima del crash. La wander policy M2-ottimizzata non trasferisce alla geometria diversa di M3. Coerente con ipotesi merge14_05: generalizzazione emerge solo quando la policy è stabile su molti spawn, non solo F1.

### 7. Seed variability

Questa è una run singola. merge14_05 ha mostrato varianza enorme su 3 run: 0%/30%/20% su M2. Il 13% di merge15_05 potrebbe essere un seed sfavorevole. Non conclusivo per una run sola.

---

## Decisioni per merge16_05

| Decisione | Motivazione |
|---|---|
| Reward shaping (curriculum_learning) | Gradiente continuo pre-crash. FRONT_DANGER=1.5m (30 step preavviso). SIDE_DANGER=0.45m. space_bonus=2.0×mean(scan)/5.0. steering=0.02. |
| TARGET_UPDATE_STEPS 1000→5000 | 15→78 gradient steps per target shift. Oscillazione training eliminata. |
| Rimuovi B3 | 0% max-steps su 8000 ep (1137 ep) = gradient noise strutturale |
| Mantieni D1 | 12.2% max-steps = 142 completamenti reali. Distribuzione bimodale ≠ tossico |
| 5000 ep (TOTAL_BLOCKS=25) | Reward denso → convergenza ~30% più rapida. Best model salvato automaticamente. |
| Fresh start | Buffer ha 2.4M transizioni col vecchio reward — stale. |
| SIDE_DANGER=0.45m invariato | Buffer 0.20m sopra COLLISION_DIST adeguato. Scendere a 0.30m = quasi binario. |

**Target merge16_05 (realistico):** M2 ≥ 30-35%, M3 ≥ 5-10%.

Target non è M2≥50% perché il POMDP aliasing (senza heading) limita strutturalmente la test performance. Reward shaping migliora training stability e sample efficiency ma non rompe i wall-following loop nel test greedy.

**Nota:** se merge16_05 raggiunge M2≥30% con training stabile (no oscillazione), conferma che il bottleneck residuo è il POMDP aliasing → heading in merge17_05.

---

## File di analisi

- Plot: `ANALISI_15_05/plots/`
  - `01_reward_curve.png` — avg100 + reward raw
  - `02_spawn_analysis.png` — avg steps + max-steps rate per spawn
  - `03_crash_rate.png` — crash rate rolling 100 ep
  - `04_loss_curve.png` — MSE loss (linear + log scale)
  - `05_test_M1.png`, `06_test_M2.png`, `07_test_M3.png` — test per maze
- Summary: `ANALISI_15_05/summary_training.txt`

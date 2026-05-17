# merge16_05 — Analisi 2 Run
**Data:** 2026-05-17  
**Branch:** `merge16_05` (da `merge15_05`)  
**Config:** 5000 ep, M2-only, reward shaping, TARGET_UPDATE=5000, 6 spawn (B3 rimosso)

---

## 1. Risultati Training

| Metrica | Run 1 | Run 2 |
|---|---|---|
| Total episodes | 5000 | 5000 |
| Crash rate totale | 80.8% | 81.1% |
| Crash rate last 100 ep | 60.0% | 67.0% |
| Final avg100 | **+197.5** | **-65.4** |
| Best avg100 | +354.3 @ ep 4945 | +360.2 @ ep 3601 |
| avg100 > 0 raggiunto | ep 2644 | ep 2652 |

### Spawn breakdown training

| Spawn | Run1 avg_steps | Run1 max-steps% | Run2 avg_steps | Run2 max-steps% |
|---|---|---|---|---|
| F2 (-1.5,-4.0) | 419.4 | **0%** ← | 415.4 | **0%** ← |
| F3 (6.0,6.0) | 399.3 | 29.5% | 422.8 | 47.5% |
| F1 (-4.5,-3.5) | 359.3 | 60.3% | 378.5 | 63.8% |
| C2 (-7.0,5.0) | 314.4 | **0%** ← | 312.0 | **0.1%** ← |
| A1 (-6.0,0.0) | 207.1 | 14.0% | 227.0 | 1.6% |
| D1 (1.5,0.0) | 143.0 | 11.9% | 66.4 | **0%** ← |

---

## 2. Risultati Test

### Maze 2 (90 ep, ε=0.0)

| Spawn | Run1 success | Run1 avg_steps | Run2 success | Run2 avg_steps |
|---|---|---|---|---|
| F1 (-4.5,-3.5) | **100%** | 500 | **100%** | 500 |
| A1 (-6.0,0.0) | **100%** | 500 | **0%** | 440 |
| F3 (6.0,6.0) | 0% | 375 | **100%** | 500 |
| D1 (1.5,0.0) | 15% | 450 | 0% | 65 |
| F2 (-1.5,-4.0) | **0%** | 450 | **0%** | 435 |
| C2 (-7.0,5.0) | **0%** | 350 | **0%** | 355 |
| **Global** | **46%** | — | **33%** | — |

### Maze 3 (90 ep, ε=0.0)

| Metrica | Run 1 | Run 2 |
|---|---|---|
| Global success | 0% | 0% |
| Avg steps | ~100 | ~75 |

---

## 3. Findings

### Finding 1 — Dead-end Exploitation (F2, C2)

**Sintomo:** F2 e C2 raggiungono avg 415-450 step in training E test, con 0% max-steps e 0% success. Sopravvivono a lungo senza mai completare.

**Meccanismo:**
- Robot spawna in zona geometricamente dead-end
- `space_bonus = 2.0 × mean(ALL 50 beams) / 5.0` premia mean(scan) alto
- In dead-end con fronte aperta: mean(scan) alto → space_bonus elevato
- Tentativo di uscita verso passaggio stretto: front_danger si attiva → penalità
- Policy ottimale sotto questa reward = **oscillare nel dead-end**, non tentare l'uscita

**Reward confronto:**
- Dead-end oscillation: `+7/step × 430 - 1000 ≈ +2010` (economicamente dominante)
- Tentativo uscita con danger: `+4/step × 200 - 1000 ≈ -200` (se fallisce)

**Confronto merge15_05 (binary reward):**
- F2 max-steps training merge15_05: ~5% (ε=0.05 salvava per caso)
- F2 max-steps training merge16_05: 0% (policy attivamente preferisce dead-end)
- Conclusione: shaping ha **peggiorato** un problema pre-esistente

### Finding 2 — Varianza Estrema tra Run (Seed Brittleness)

| Spawn | Run1 test | Run2 test |
|---|---|---|
| A1 (-6.0,0.0) | 100% | **0%** |
| F3 (6.0,6.0) | 0% | **100%** |

La policy memorizza 1-2 traiettorie specifiche che cambiano tra run. Non è generalizzazione — è **trajectory memorization** seed-dipendente. Henderson et al. 2018 ("Deep Reinforcement Learning That Matters"): variance across seeds in deep RL può essere dell'ordine di grandezza dei risultati stessi.

### Finding 3 — Policy Degradation Run2

Run2: best avg100 = 360.2 @ ep 3601 → final avg100 = -65.4 @ ep 5000. Regressione di 425 punti in 1400 episodi. TARGET_UPDATE=5000 ha ridotto l'oscillazione tipica di merge15_05 (±800 pts) ma non ha eliminato la degradazione a lungo termine.

### Finding 4 — Instabilità Cronica

Entrambe le run: dopo ep 2650, avg100 oscilla [-100, +300] senza convergenza. Training non stabilizzato a 5000 ep. Causa: reward landscape con local optima (dead-end attractors) che il DDQN continua a esplorare.

---

## 4. Analisi Causale — Reward Shaping e Overfitting

### Terminologia corretta

Il fenomeno osservato NON è "overfitting" nel senso ML classico (memorizzazione dati di training). È una combinazione di:
1. **Reward hacking su local optimum** (F2, C2): la policy massimizza correttamente la reward shaped, ma quella reward non è allineata col task
2. **Trajectory memorization** (F1 sempre 100%, A1/F3 che cambiano): POMDP aliasing + training spawn limitati → policy specializzata per traiettorie specifiche
3. **Seed brittleness**: varianza RL standard con training non convergito

### Correlazione Reward Shaping → Dead-end Exploitation

**Ng et al. 1999** garantisce invarianza della policy ottimale SOLO per shaping potential-based: `F(s,a,s') = γΦ(s') - Φ(s)`.

`space_bonus = 2.0 × mean(scan) / 5.0` è **NON potential-based**: è una reward additiva sull'istante corrente senza funzione potenziale verso un goal. Viola la condizione di Ng.

**Devlin & Kudenko 2012** ("Potential-Based Difference Rewards"): shaping non-potential crea local optima spurii. Il dead-end con fronte aperta è esattamente un local optimum spurio: non esiste nella reward binaria (+5 ovunque), esiste con space_bonus (open facing → high bonus).

**Causa primaria (indipendente dal shaping):** POMDP aliasing — Mirowski et al. 2016: senza heading nello stato, il robot non distingue "corridoio stretto verso exit" da "dead-end con vista aperta". LIDAR identico → stessa azione.

**Causa secondaria (specifica del shaping):** `mean(ALL beams)` include raggi laterali e posteriori. In dead-end orientato verso spazio aperto, mean(scan) è alto indipendentemente dall'orientamento di navigazione. Fix: usare solo settore frontale.

### Assenza di Goal Representation

**Tai et al. 2017** ("Virtual-to-real Deep RL for Mapless Navigation"): reward binaria funziona con goal esplicito perché il goal crea gradiente direzionale. **Zhu et al. 2017** ("Target-Driven Visual Navigation"): senza goal nello stato, l'agente ottimizza sopravvivenza, non navigazione.

Il nostro sistema non ha goal → space_bonus sostituisce parzialmente il gradiente direzionale, ma lo fa in modo non-potential → local optima. Questo è il problema fondamentale che né il reward shaping né più episodi risolvono.

---

## 5. Spawn da Rimuovere

Basato su dati training + test:

| Spawn | Verdetto | Motivazione |
|---|---|---|
| F2 (-1.5,-4.0) | **RIMUOVERE** | 0% max-steps training (entrambe le run), 0% success test (entrambe le run), avg 430 step = dead-end confermato strutturale |
| C2 (-7.0,5.0) | **RIMUOVERE** | 0-0.1% max-steps training, 0% success test (entrambe le run), avg 330-355 step = dead-end |
| D1 (1.5,0.0) | **MANTENERE** | Run1: 12% max-steps training, 15% success test. Run2: 0% (seed brittleness). Non strutturalmente tossico. |

Dopo rimozione: **4 spawn training M2** (F1, F3, A1, D1).

---

## 6. Note su Gemini Analysis

Gemini ha identificato correttamente: seed brittleness (Henderson 2018), policy degradation, failure di generalizzazione. Tuttavia:
- Cifre per F3 (44.9%) e C2 (38.6%) non coerenti con i plot (0% per entrambi in run1). Possibile errore di lettura CSV o dato da run diversa.
- Non identifica il meccanismo dead-end exploitation.
- Raccomanda multi-maze training (corretto) ma non specifica il problema di space_bonus.

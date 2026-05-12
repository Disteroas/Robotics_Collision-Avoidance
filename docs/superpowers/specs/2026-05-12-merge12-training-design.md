# merge12_05 — Training Design

**Data:** 2026-05-12  
**Branch target:** `merge12_05` (da `merge11_05`)  
**Autore:** Davide Covolo + Claude

---

## Obiettivo

Correggere i bug identificati in `merge11_05` che hanno causato M2=0% nel test. Il training di merge12_05 è identico a merge11_05 tranne per 3 fix:

1. `MAX_STEPS`: 1000 → 500 (fix critico — causa root di M2=0%)
2. Blocchi: 25×200 → 45×100 ep (riduce finestra catastrophic forgetting)
3. `best_avg` persistito in checkpoint (fix bug)

Nessuna altra variabile viene cambiata, per mantenere la causalità isolata.

---

## Analisi merge11_05 — perché ha fallito

### Risultati test

| Maze | Success rate | Avg reward | Baseline (randomSpawn 05_08) |
|------|-------------|-----------|-------------------------------|
| M1 | 57% (17/30) | ~+1193 | 30% |
| M2 | **0% (0/30)** | ~+215 | 33% |
| M3 | **0% (0/30)** | ~-577 | 33% |

### Causa root — MAX_STEPS mismatch

Durante training merge11_05: `MAX_STEPS = 1000`. In test: `MAX_STEPS = 500`.

Con `MAX_STEPS=1000` e 16 spawn random su M2 (labirinto_9b), un episodio "successo" richiede che il robot sopravviva 1000 step senza collisioni — improbabile in un maze complesso con molti ostacoli. La quasi totalità degli episodi M2 in training termina con crash → reward sempre negativa → nessun segnale positivo per imparare la navigazione corretta.

**Evidenza empirica:**
- randomSpawn 05_08 (MAX_STEPS=500, M2 solo): training success M2 ~52%, test M2 = 33%
- merge11_05 (MAX_STEPS=1000, M2 interleaved): training success M2 ~0%, test M2 = 0%

**Motivazione teorica (Tobin et al. 2017):** con N spawn diversi, la copertura dello spazio di stato è proporzionale a N×T (spawn × step per episodio). Con N=16 e T=500, la copertura è già molto più ricca di quella ottenibile con spawn fisso. Aumentare T oltre 500 non migliora la copertura ma rende M2 più difficile da "completare", degradando il segnale di apprendimento.

**Motivazione teorica (GAMMA=0.99):** l'orizzonte effettivo del discounting è 1/(1-γ) = 100 step. Un episodio di 500 step equivale a 5 orizzonti completi — ampiamente sufficiente per propagare il valore.

### M1 analisi spawn

Il test di M1 rivela comportamento bimodale:
- **P1 (-2.9,-2.0,N):** 17/17 successi — canale sinistro, traiettoria lineare verso N
- **P2 (1.0,-1.0,N):** 0/13 successi, crash step ~97-98 — **P2 mantenuto per diagnosi**

P2 è una trappola geometrica o una posizione che l'agente non ha ancora imparato a gestire. Mantenerla nel training è la scelta corretta: forzare l'agente a imparare anche questa posizione difficile. La diagnosi sarà: se dopo merge12_05 P2 crasha ancora allo stesso step → trappola geometrica; se crash step varia → sta imparando ma non ancora convergito.

### Bug best_avg

`best_avg = -float('inf')` viene inizializzato in `train.py` senza essere caricato dal checkpoint. Ad ogni nuovo blocco, `best_avg` riparte da `-inf`. Questo significa che il modello salvato come "best" durante il blocco N viene confrontato con `-inf` all'inizio del blocco N+1 — il primo episodio del blocco N+1 (con avg100 tipicamente negativa) potrebbe sovrascrivere il modello migliore.

---

## Decisioni di design

### 1. MAX_STEPS = 500 (fix critico)

Training e test usano lo stesso orizzonte. Nessuna altra modifica.

```python
# train.py:41
MAX_STEPS = 500
```

### 2. Blocchi 100 ep (era 200)

45 blocchi × 100 ep = 4500 ep totali.

**Motivazione:** blocchi più corti riducono la finestra di catastrophic forgetting (McCloskey & Cohen 1989; Goodfellow et al. 2013). Con blocchi da 200 ep, il modello può "dimenticare" M1 durante 400 ep consecutive di M2. Con blocchi da 100 ep, la finestra massima è 200 ep.

**4500 vs 5000 ep:** riduzione del 10%. Accettabile — la curva ε raggiunge ancora il minimo (~0.05) entro ep 3000.

Calcolo ε:
```
ε(ep) = max(0.05, 1.0 × 0.999^ep)
ep 1000: ε = 0.368
ep 2000: ε = 0.135
ep 3000: ε = 0.050  ← minimo naturale
ep 4500: ε = 0.050  (clamped)
```

### 3. best_avg nel checkpoint

```python
# train_core.py — save_ckpt:
data = {
    ...
    'best_avg': best_avg,   # aggiungere
}

# train_core.py — load_ckpt:
best_avg = d.get('best_avg', -float('inf'))
return ep, crashes, best_avg   # aggiungere best_avg al return

# train.py — main:
ep0, crashes, best_avg = load_ckpt(agent, args.checkpoint, rh)
```

### 4. Pattern M1/M2/M2 invariato

Pattern (M1, M2, M2) ciclico, ratio 1:2. Motivazione (Cobbe et al. 2019): diversità ambienti è il principale predittore di generalizzazione in DRL. M2 riceve 2× più episodi perché è il maze più difficile e non ancora convergito.

### 5. Reward invariata

Reward: `+5` per step sopravvissuto, `-1000` per collisione (termine episodio).

```python
# usv_logic.py:compute_reward (invariato)
if float(np.min(scan)) < COLLISION_DIST:
    return -1000.0, True
return 5.0, False
```

**Motivazione:** reward shaping è il passo successivo se merge12_05 non generalizza su M3. Non aggiungere variabili confondenti prima di avere la baseline corretta.

**Nota su action_index:** il parametro `action_index` in `compute_reward` è dead code (accettato ma mai usato). Non rimuovere in questo branch — non è blocking e rimuoverlo aggiunge una variabile al confronto.

### 6. SPAWN_LISTS invariate

M1: P1 e P2 (entrambi mantenuti).  
M2: 16 punti (zone A-F, validati con min_lidar ≥ 0.43m).  
TEST_SPAWN_LISTS: M1={P1,P2}, M2={A1,B3,C2,D2,E2,F1}, M3={(-2,-1,0)}.

### 7. Hyperparametri invariati

| Parametro | Valore | Note |
|-----------|--------|------|
| BATCH_SIZE | 64 | DQN standard |
| LR | 0.00025 | |
| GAMMA | 0.99 | Orizzonte 100 step |
| BETA_DECAY | 0.999 | ε → 0.05 a ep ~3000 |
| MSELoss | ✓ | Confermato corretto (ADR-06) |
| clip=10.0 | ✓ | Confermato corretto (ADR-08) |
| TARGET_UPDATE | 1000 steps | |
| MEMORY_CAPACITY | 100K | |
| GAZEBO_SPEED | 5× | Confermato stabile |

---

## File da modificare

| File | Modifica | Dettaglio |
|------|---------|-----------|
| `src/my_usv/scripts/train.py` | `MAX_STEPS` 1000→500; docstring; firma `load_ckpt` | Critico |
| `src/my_usv/scripts/train_core.py` | `best_avg` in `save_ckpt`/`load_ckpt`; return type | Importante |
| `start_train_multimaze.sh` | `TOTAL_BLOCKS=45`, `BLOCK_SIZE=100`, header comment | Critico |

**Invariati:** `usv_env.py`, `usv_logic.py`, `ddqn_model.py`, `test.py`, `start_test.sh`, `train_core.py` (hyperparametri)

---

## Validazione pre-training

1. `git checkout -b merge12_05` (da `merge11_05`)
2. Applicare i 3 fix
3. Verificare: `grep "MAX_STEPS" src/my_usv/scripts/train.py` → deve dare `500`
4. Verificare: `grep "TOTAL_BLOCKS\|BLOCK_SIZE" start_train_multimaze.sh` → `45`, `100`
5. Avviare: `./start_train_multimaze.sh --reset`

---

## Metriche di successo

| Metrica | Target | merge11_05 | randomSpawn baseline |
|---------|--------|-----------|----------------------|
| Test M1 success rate | ≥ 90% | 57% | 30% |
| Test M2 success rate | ≥ 50% | 0% | 33% |
| Test M3 success rate | ≥ 30% | 0% | 33% |
| Training avg100 finale | ≥ 1500 | N/D | N/D |

**Diagnosi P2 M1:** se dopo training P2 crasha ancora a step ~97-98 → trappola geometrica (rimuovere P2 da SPAWN_LISTS[1]). Se crash step varia → apprendimento in corso, aumentare training.

---

## Passo successivo (se merge12_05 non generalizza M3)

- **Reward shaping** (Ng et al. 1999): aggiungere danger penalty graduata basata su min_lidar
- **NON fare:** Huber, clip=1.0, PER (tutti testati e peggiorano — vedi ANALISI_FIXED_FENG_FALLIMENTO.md)

---

## Riferimenti

- Mnih et al. (2015) — DQN: epsilon annealing, batch size, target network
- Cobbe et al. (2019) — Quantifying Generalization in RL: environment diversity
- Tobin et al. (2017) — Domain Randomization: spawn/initial condition randomization
- McCloskey & Cohen (1989) — Catastrophic Interference: forgetting in neural nets
- Goodfellow et al. (2013) — Catastrophic Forgetting: benchmark e analisi
- Ng et al. (1999) — Policy Invariance Under Reward Transformations: reward shaping
- Feng et al. (2021) — Paper baseline: DDQN collision avoidance USV

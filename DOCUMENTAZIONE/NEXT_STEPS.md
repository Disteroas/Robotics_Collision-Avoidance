# Next Steps — backlog tecnico prioritizzato

**Aggiornato 2026-05-17** — analisi merge16_05 (2 run) completata.  
**Branch corrente di sviluppo:** `matte_merge17_05`

---

## Stato della conoscenza

### Progressione esperimenti

| Branch | Episodi | M2 test | M3 test | Fenomeno |
|---|---|---|---|---|
| `merge12_05` | 4500 | 46.7% | 0% | Prima convergenza stabile |
| `merge14_05` run3 | 4000 | 20% | **13%** | Unica generalizzazione M3 osservata |
| `merge15_05` | 8000 | 13% | 0% | Overfitting posizionale, instabilità |
| `merge16_05` run1 | 5000 | 46% | 0% | Dead-end exploitation confermato |
| `merge16_05` run2 | 5000 | 33% | 0% | Policy degradation, seed brittleness |

### Ipotesi di lavoro consolidate (NON cambiare)

- MSE loss + clip=10.0: corretto (Huber + clip=1.0 testato → fallito in `fixed_feng`)
- Uniform replay: corretto (PER testato → peggiora, Feng 2021 §3.2)
- BETA_DECAY=0.999: corretto (ε→0.05 a ep ~3000, orizzonte appropriato)
- GAMMA=0.99: corretto (orizzonte 100 step; GAMMA=0.999 → orizzonte 1000 > MAX_STEPS → Q esplodono)
- REPLAY_START_SIZE=10,000: corretto (Mnih 2015 — buffer diversificato prima del primo gradient step)

---

## Findings chiave merge16_05

### 1. Dead-end Exploitation — meccanismo confermato

F2 (-1.5,-4.0) e C2 (-7.0,5.0): 0% max-steps training, 0% success test, avg 330-450 step in ENTRAMBE le run. Confermato strutturale.

**Meccanismo:** `space_bonus = 2.0 × mean(ALL 50 beams) / 5.0` premia open space indipendentemente dalla direzione. In dead-end con fronte aperta: mean(scan) alto → bonus alto → policy preferisce oscillare nel dead-end piuttosto che tentare exit stretto (dove front_danger attiva penalità).

**Reward hacking:** la policy massimizza correttamente la shaped reward, ma quella reward crea local optimum spurio non allineato col task (navigazione). Ng et al. 1999 garantisce invarianza solo per shaping potential-based — la nostra reward non lo è.

**Confronto binary vs shaped per F2 (training):**
- merge15_05 (binary): F2 max-steps 5% — ε=0.05 occasionalmente usciva per caso
- merge16_05 (shaped): F2 max-steps 0% — policy sistematicamente **preferisce** il dead-end

### 2. Causa primaria: POMDP aliasing

Mirowski et al. 2016: senza heading nello stato, LIDAR-only crea aliasing tra "corridoio stretto verso exit" e "dead-end orientato verso spazio aperto". Stesso vettore LIDAR → stessa azione. Reward shaping peggiora questo problema (punto 1) ma non ne è la causa radice.

### 3. Assenza di goal representation

Tai et al. 2017, Zhu et al. 2017: reward binaria funziona per collision avoidance WITH goal (goal crea gradiente direzionale). Senza goal, space_bonus sostituisce il gradiente direzionale in modo non-potential → local optima. Questo è il bottleneck fondamentale che né reward shaping né più episodi risolvono strutturalmente.

### 4. Seed Brittleness

Henderson et al. 2018: varianza across seeds in deep RL dell'ordine di grandezza dei risultati. Esempio: A1 → 100% run1, 0% run2; F3 → 0% run1, 100% run2. La policy memorizza 1-2 traiettorie specifiche seed-dipendenti, non una policy generale.

### 5. TARGET_UPDATE=5000 — effetto parziale

Ha ridotto oscillazione ±800 pts di merge15_05. Non ha eliminato policy degradation run2 (360→-65 in 1400 ep). Cause residue: reward landscape con dead-end attractors (reward hacking), seed brittleness.

---

## Roadmap merge17_05

### Azioni NECESSARIE (bugs/regressioni Matteo)

**A. Ripristino min-pooling LIDAR** (`usv_logic.py`)

```python
# REVERTIRE: uniform sampling di Matteo introduce blind spots 5.3° tra campioni
indices = np.linspace(0, len(scan) - 1, n_bins, dtype=int)
return scan[indices]

# RIPRISTINARE: min-pooling (garantisce visibilità ostacolo in ogni bin angolare)
chunks = np.array_split(scan, n_bins)
return np.array([np.min(chunk) for chunk in chunks])
```

Motivazione: Mnih 2015 DQN usa max-pooling (equivalente per presence detection); Tai et al. 2017 usa min-pooling per collision avoidance. Uniform sampling lascia gap angolari ciechi → aumento crash strutturale.

**B. TARGET_UPDATE_STEPS = 5000** (`train_core.py`)

```python
TARGET_UPDATE_STEPS = 5_000   # era 1_000 su branch Matteo (regressione)
```

Motivazione: Van Hasselt et al. 2016 (DDQN): target net deve essere stabile durante l'ottimizzazione della policy. Con 1000 steps e MAX_STEPS=500: ~15 gradient steps per target shift → instabilità confermata in merge15_05 (oscillazione ±800 pts). Con 5000: ~78 gradient steps → convergenza locale prima del prossimo shift.

**C. Reward shaping da merge16_05** (`usv_logic.py`)

Portare tutto il reward shaping di merge16_05 (FRONT_DANGER=1.5m, SIDE_DANGER=0.45m, space_bonus, steering penalty). Non tornare a binary reward.

**D. Rimuovere F2 e C2 da SPAWN_LISTS[2]** (`usv_env.py`)

```python
2: [
    (-6.0,  0.0,  0.0  ),  # A1
    (-4.5, -3.5,  0.0  ),  # F1
    (-1.5, -4.0,  1.571),  # F2  ← RIMUOVERE (dead-end strutturale, 0% entrambe le run)
    ( 6.0,  6.0,  3.142),  # F3
    (-7.0,  5.0,  0.0  ),  # C2  ← RIMUOVERE (dead-end strutturale, 0% entrambe le run)
    ( 1.5,  0.0,  3.142),  # D1
],
```

Motivazione: F2 e C2 — 0% max-steps training (entrambe le run), 0% success test (entrambe le run), avg 330-450 step = dead-end confermato. Aggiungono noise al training senza contribuire a policy utili. Simile a B3 (rimosso in merge16_05 per stessa ragione).

**E. Fix --total-ep e docstring** (`train.py`)

```python
p.add_argument('--total-ep', type=int, default=5000)  # era 4000
# docstring: "Con BETA_DECAY=0.999 e 5000 episodi:"
```

**F. Allineamento TEST_SPAWN_LISTS[2] = SPAWN_LISTS[2]** (`usv_env.py`)

Già fatto in merge16_05. Portare la modifica su matte_merge17_05.

### Azioni RACCOMANDATE (miglioramenti nuovi)

**G. Fix space_bonus — solo settore frontale** (`usv_logic.py`)

```python
# PROBLEMA ATTUALE: mean(ALL 50 beams) premia dead-end orientati verso spazio aperto
space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE

# FIX PROPOSTO: mean(solo settore frontale bins 15-35)
space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan[15:35])) / LIDAR_MAX_RANGE
```

Rationale: in dead-end con fronte aperta e robot che ha girato → il settore frontale punta verso la parete (low scan) → space_bonus basso → no attractor. In corridoio aperto verso exit → front scan alto → space_bonus alto. Allineamento reward-task migliore.

Alternativa più conservativa: mantenere mean(ALL beams) ma ridurre weight da 2.0 a 1.0 per ridurre attrazione dead-end senza cambio architetturale.

**H. seed strategy corretta** (se mantenere set_seed)

Se si vuole riproducibilità, set_seed deve essere **solo in test** (ε=0.0 già deterministico condizionato al seed). In training, seed fisso introduce bias sistematico nella sequenza di esplorazione per ogni blocco (ogni blocco resetta RNG a seed 42). Alternativa: usare seed diversi per run diverse (42, 123, 0) e confrontare risultati — Henderson et al. 2018 raccomanda ≥5 seed per claim robusti.

```python
# test.py — OK: rende la valutazione deterministica
set_seed(42)

# train.py — RIMUOVERE o parametrizzare:
# set_seed(42)  ← rimosso, o passare seed come --seed argomento CLI
```

**I. Multi-maze training** (opzionale ma raccomandato)

Cobbe et al. 2019: test set misura generalizzazione solo se diverso da training. M3 = 0% in tutte le run → zero generalizzazione. Multi-maze (M1+M2 interleaved) con pattern 1:2 ha mostrato M2=46.7% in merge12_05 — paragonabile a best M2-only run.

Gemini e la letteratura concordano: Cobbe et al. 2019, Zhao et al. 2020 ("Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics"): diversità degli ambienti di training è requisito necessario per generalizzazione. Training M2-only produce specialista locale — confermato da tutti gli esperimenti.

### Azioni DIFFERITE (future iterazioni)

**J. Heading nello stato** (`usv_env.py`, `ddqn_model.py`)

```python
# Aggiungere [cos(yaw), sin(yaw)] al vettore stato: 50 → 52 dim
state = np.concatenate([lidar_50bins, [cos(yaw), sin(yaw)]])
```

Motivazione: Mirowski et al. 2016, Tai et al. 2017: heading risolve POMDP aliasing per collision avoidance. Senza heading il robot non distingue "mi sto avvicinando" da "mi sto allontanando" → wall-following loop deterministico in test (ε=0.0). Richiede ristrutturazione modello (STATE_DIM: 50→52).

**K. Goal representation** (`usv_env.py`, `ddqn_model.py`)

```python
state = np.concatenate([lidar_50bins, [dist_to_goal, cos(angle_goal), sin(angle_goal)]])
# 50 → 53 dim
```

Motivazione: Zhu et al. 2017, Tai et al. 2017. Richiede definizione goal fisso per maze, lettura posizione robot da Gazebo (`/gazebo/model_states` topic). Oltre scope Feng 2021 (che fa pure collision avoidance senza goal).

---

## Letteratura di riferimento

### Reward shaping e local optima

- **Ng et al. 1999** — "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping". ICML. Condizione necessaria e sufficiente per invarianza della policy ottimale: `F = γΦ(s') - Φ(s)`. Qualsiasi shaping non-potential può cambiare la policy ottimale.
- **Devlin & Kudenko 2012** — "Dynamic Potential-Based Reward Shaping". AAMAS. Estende Ng a ambienti non stazionari. Dimostra che shaping non-potential crea local optima spurii. Il nostro space_bonus rientra in questa categoria.
- **Grzes & Kudenko 2009** — "Theoretical and Empirical Analysis of Reward Shaping in Reinforcement Learning". Shaping può ridurre convergenza del 30-60% ma non-potential shaping modifica la policy ottimale in modo imprevedibile.

### Varianza e seed brittleness

- **Henderson et al. 2018** — "Deep Reinforcement Learning That Matters". AAAI. Documenta varianza estrema tra seed in deep RL. Claim: varianza tra seed può essere dell'ordine di grandezza dei risultati stessi. Raccomanda ≥5 seed per claim statisticamente robusti.

### Collision avoidance e navigazione

- **Tai et al. 2017** — "Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation". IROS. Usa LIDAR (10 beam) + relative goal position → reward binaria. Dimostra che binary reward funziona per collision avoidance WITH goal.
- **Mirowski et al. 2016** — "Learning to Navigate in Complex Environments". ICLR. Aggiunge heading e depth frame allo stato. Senza heading: POMDP aliasing → wall-following loop. Con heading: policy stabile e generalizzabile.
- **Zhu et al. 2017** — "Target-Driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning". ICRA. Goal representation necessaria per navigazione direzionale. Senza goal: agente ottimizza sopravvivenza, non navigazione.

### DDQN e target network

- **Van Hasselt et al. 2016** — "Deep Reinforcement Learning with Double Q-learning". AAAI. Target network deve essere aggiornata ogni N gradient steps sufficienti per convergenza locale. Con step troppo frequenti: policy oscilla inseguendo target instabile.
- **Mnih et al. 2015** — "Human-level control through deep reinforcement learning". Nature. REPLAY_START_SIZE: buffer deve essere diversificato (random exploration) prima del primo gradient step. Uniform replay corretto per Q-learning standard.

### Test set methodology

- **Cobbe et al. 2019** — "Quantifying Generalization in Reinforcement Learning". ICML. Test set misura generalizzazione solo se diverso dal training set. TEST_SPAWN_LISTS deve essere identico a SPAWN_LISTS per M2 (misura policy quality), diverso per M3 (misura zero-shot generalization).

### Transfer e multi-environment

- **Pan & Yang 2010** — "A Survey on Transfer Learning". IEEE TKDE. Negative transfer: training su ambiente sorgente può degradare performance sull'ambiente target se le distribuzioni differiscono troppo. Training M1+M2 senza bilanciamento corretto → M2 policy degradata da M1.
- **Zhao et al. 2020** — "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey". Diversità ambienti di training = requisito necessario per generalizzazione. Specializzazione su singolo maze → performance zero su maze unseen.

---

## Decisioni irreversibili (NON modificare)

| Decisione | Motivazione | Ref |
|---|---|---|
| MSE loss + clip=10.0 | Testato: Huber+clip=1.0 → avg100<0 in `fixed_feng` | `fixed_feng` analysis |
| Uniform replay | Testato: PER peggiora reward finale | Feng 2021 §3.2 |
| GAMMA=0.99 | GAMMA=0.999 → orizzonte 1000 step > MAX_STEPS → Q explode | Van Hasselt 2016 |
| NON usare `fixed_feng` come base | Config errata confermata | `ANALISI_FIXED_FENG_FALLIMENTO.md` |

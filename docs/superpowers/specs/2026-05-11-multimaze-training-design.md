# Multi-Maze Interleaved Training Design

**Data:** 2026-05-11  
**Branch target:** `merge11_05`  
**Autore:** Davide Covolo + Claude

---

## Obiettivo

Training DDQN su Maze 1 e Maze 2 in modalità interleaved (1:2 ratio) con random spawn in entrambi i maze. Obiettivo: ottenere buona performance su M1/M2 **e** generalizzazione su M3 (test-only, mai visto in training).

---

## Analisi delle run precedenti

| Run | Maze | Spawn | Reward | ε-decay | Test M1 | Test M2 | Test M3 |
|-----|------|-------|--------|---------|---------|---------|---------|
| multi_maze 05_01 | M1+M2 interleaved | fisso | complessa | 0.995 | 100% | 77% | 0% |
| curriculum_learning 05_06 | M1→M2 curriculum | fisso | complessa | 0.995 | 100% | 0% | 0% |
| randomSpawn 05_08 | M2 only | random | binaria+penalty | 0.999 | 30% | 33% | 33% |

**Cause radice identificate:**
- multi_maze: spawn fisso → memorizza percorsi specifici → 0% M3
- curriculum_learning: ε=0.05 a ep ~600 con BETA_DECAY=0.995 + catastrophic forgetting (McCloskey & Cohen 1989; Goodfellow et al. 2013)
- randomSpawn: random spawn → policy non path-specific → generalizzazione M3. Ma single-maze → performance assoluta limitata

**Ipotesi di lavoro:** combinare multi-maze (performance) + random spawn (generalizzazione) + BETA_DECAY=0.999 (esplorazione corretta) = migliore compromesso.

---

## Decisioni di design

### 1. Training interleaved M1:M2 = 1:2

Pattern blocchi da 200 episodi:
```
M1(200) → M2(200) → M2(200) → M1(200) → M2(200) → M2(200) → ...
```

**Motivazione (Cobbe et al. 2019):** diversità ambienti è il principale predittore di generalizzazione in DRL. Il ratio 1:2 bilancia:
- M1 fornisce segnale stabile early training + diversità topologica
- M2 (più difficile) riceve 2× più episodi
- MSE loss auto-pesa: TD-error crash M2 (~1000²) >> episodio M1 tranquillo (~10²), quindi M2 domina il gradiente anche con 50% meno episodi

Nessun curriculum, nessuna phase transition. Gazebo riavvia ogni 200 episodi per caricare il world file corrispondente.

### 2. Random spawn entrambi i maze

**M2:** 16 punti già validati (zone A-F, min_lidar ≥ 0.43m). Invariati.

**M1:** espandere da 8 a 16 punti (zone A-F analoghe). I punti proposti:

```python
SPAWN_LISTS = {
    1: [
        # Zone A: sud (3 spawn)
        (-3.0, -5.0,  1.57),   # A1: heading N
        ( 0.0, -4.5,  1.57),   # A2: centre-south heading N
        ( 2.5, -5.0,  1.57),   # A3: right-south heading N
        (-1.5, -5.0,  0.0 ),   # A4: south heading E

        # Zone B: sinistra (3 spawn)
        (-2.9, -2.0,  1.57),   # B1: left channel entry heading N
        (-2.9,  0.5,  0.0 ),   # B2: left channel mid heading E
        (-2.5, -3.5,  0.785),  # B3: left-mid heading NE

        # Zone C: centro (3 spawn)
        ( 0.5, -2.5,  1.57),   # C1: centre-bottom heading N
        ( 0.0, -1.0,  3.142),  # C2: centre heading W
        ( 0.0,  0.0,  1.57),   # C3: centre heading N

        # Zone D: destra (2 spawn)
        ( 2.5, -2.0,  1.57),   # D1: right outer heading N
        ( 2.5,  0.0,  3.142),  # D2: right heading W

        # Zone E: superiore (2 spawn)
        (-1.0,  1.5,  0.0 ),   # E1: upper-left heading E
        ( 1.0,  1.5,  3.142),  # E2: upper-right heading W

        # Zone F: extra coverage (2 spawn)
        (-3.0, -4.0,  0.0 ),   # F1: south-left heading E
        ( 1.5, -4.0,  1.57),   # F2: south-right heading N
    ],
    ...
}
```

⚠️ **I punti M1 vanno validati con `./test_spawns.sh 1` prima del training.** Il safety check esistente (min_lidar ≥ 0.40m, 3 retry) è il fallback ma non garantisce sicurezza.

**Motivazione (Tobin et al. 2017 — domain randomization):** randomizzare condizioni iniziali produce policy robuste. Dati empirici interni: randomSpawn con spawn random M2 → 33% M3 vs multi_maze spawn fisso → 0% M3.

### 3. Epsilon decay globale

BETA_DECAY=0.999 (già in `train_core.py`). Epsilon **mai resettato** al cambio maze.

```
ε(ep) = max(0.05, 1.0 × 0.999^ep)
ep 1000: ε = 0.368
ep 2000: ε = 0.135
ep 3000: ε = 0.050  ← minimo naturale
ep 5000: ε = 0.050  (clamped)
```

**Motivazione (Mnih et al. 2015):** epsilon deve decadere per l'intera durata del training. Con BETA_DECAY=0.995, ε=0.05 a ep ~600 — 80% del training in pura exploitation su policy non convergita.

### 4. Reward function invariata

Reward complessa da `usv_logic.py` (da multi_maze 05_01): +5/step base, danger penalty front/side, -1000 crash. **Non modificata.**

**Motivazione:** multi_maze con reward complessa → 77% M2, 100% M1. La reward funziona — il bottleneck era spawn fisso e BETA_DECAY. Non aggiungere variabili confondenti.

Nota: la relazione tra tipo di reward e generalizzazione M3 è ancora una variabile confusa (feng_direct usava reward diversa). Reward shaping è il **passo successivo** se questo training non generalizza su M3 (Ng et al. 1999).

### 5. Hyperparametri invariati

Tutto `train_core.py` rimane invariato:

| Parametro | Valore | Note |
|-----------|--------|------|
| BATCH_SIZE | 64 | DQN standard (Mnih 2015: 32, noi 64) |
| LR | 0.00025 | Calibrato per BATCH=64 (Goyal 2017 scaling rule) |
| GAMMA | 0.99 | Orizzonte 100 step |
| MSELoss | ✓ | Confermato corretto (ADR-06) |
| clip=10.0 | ✓ | Confermato corretto (ADR-08) |
| TARGET_UPDATE | 1000 steps | Invariato |
| MEMORY_CAPACITY | 100K | Invariato |

### 6. Episodi e velocità

- **Totale:** 5000 episodi (training curves non convergite a 3000)
- **MAX_STEPS:** 1000 (training), 500 (test) — invariati
- **Gazebo speed:** 5× (testato e stabile su setup headless)

---

## File modificati

| File | Tipo modifica | Dettaglio |
|------|--------------|-----------|
| `usv_env.py` | Modifica | `SPAWN_LISTS[1]`: 8 → 16 punti |
| `train.py` | Modifica | Rimuovi phase transition logic; `total_ep` da CLI (default 5000) |
| `start_train_multimaze.sh` | Nuovo | Script blocchi 200ep, pattern M1/M2/M2, 5000 ep totali |

**Invariati:** `train_core.py`, `ddqn_model.py`, `usv_logic.py`, `test.py`, `start_test.sh`

---

## Validazione pre-training

1. `git checkout -b merge11_05`
2. Applicare modifiche
3. `./test_spawns.sh 1` — validare tutti 16 punti M1 (min_lidar ≥ 0.40m)
4. Rimuovere/sostituire punti non sicuri
5. Avviare training: `./start_train_multimaze.sh`

---

## Metriche di successo

| Metrica | Target | Baseline (migliore precedente) |
|---------|--------|-------------------------------|
| Test M1 success rate | ≥ 90% | 100% (multi_maze) |
| Test M2 success rate | ≥ 70% | 77% (multi_maze) |
| Test M3 success rate | ≥ 30% | 33% (randomSpawn) — da migliorare |
| Training avg100 finale | ≥ 1500 | 2034 (multi_maze) |

---

## Riferimenti

- Mnih et al. (2015) — DQN: epsilon annealing, batch size, target network
- Cobbe et al. (2019) — Quantifying Generalization in RL: environment diversity
- Tobin et al. (2017) — Domain Randomization: spawn/initial condition randomization
- Ng et al. (1999) — Policy Invariance Under Reward Transformations: reward shaping (passo futuro)
- Goodfellow et al. (2013) — Catastrophic Forgetting: perché curriculum fallisce senza replay
- Goyal et al. (2017) — Large Minibatch SGD: LR scaling con batch size
- Feng et al. (2021) — Paper baseline: DDQN collision avoidance USV

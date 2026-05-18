# Briefing 2026-05-18 — Stato del progetto e piano DDQN

**Branch corrente:** `matte_merge17_05`  
**Autori attivi:** Davide Covolo, Matteo Bolo (BoloM03)  
**Orizzonte:** 2 settimane, progetto didattico DRL

---

## 1. Commit di oggi (Matteo) — cosa è cambiato

### `3301eb1` — Replica paper + uniform sampling + seed
- LIDAR: min-pooling → **uniform sampling** (regressione, blind spots 5.3°)
- `set_seed(42)` aggiunto in `train.py` e `test.py`
- `TARGET_UPDATE_STEPS = 1_000` (regressione da merge16_05)

### `7379e15` — Soft update + prima run
- `TARGET_UPDATE_STEPS` → **soft update TAU=0.005** (wrong direction, vedi §3)
- Training lanciato: 4000 ep, crash rate **90.6%**, final avg100=709.7, best=1013 @ ep3350
- Test M2: **40% global** (A1=100%, F1≈48%, C2=0%, E2=0%, D2=0%)
- F2 e C2 ancora presenti → dead-end confermato per terza volta
- Scritto `Analisi_softupdate_and_buffer.md` che consiglia PER → **contraddice Feng 2021 §3.2**

### `6b93072` — Maze 2 aggiornato
- Pareti ridimensionate e riposizionate → dead-end geometrici rimossi
- Model rinominato `Muri9b`
- **Tutti gli spawn point richiedono ri-validazione** (clearance ≥ 0.40m)
- Confronti storici (merge12-16) non più comparabili per M2

### `f15e1a4` — Ripristino hard update + fix world
- `TARGET_UPDATE_STEPS = 5000` ✓ ripristinato (come merge16_05)
- `labirinto_9b.world`: aggiunto plugin `gazebo_ros_state` (necessario per teleport)
- Maze geometry da `6b93072` rimane in place

**Stato attuale del codice:**

| Componente | Stato | Corretto? |
|---|---|---|
| TARGET_UPDATE_STEPS | 5000 | ✓ |
| LIDAR processing | uniform sampling | ✗ → serve min-pooling |
| Reward | +5/-1000 binaria | ✓ (ma manca shaping di merge16_05) |
| set_seed(42) in train | presente | ✗ → bias esplorazione |
| Spawn LISTS[2] | 10 spawn inclusi F2/C2 | ✗ → dead-end ancora dentro |
| Frame stacking | assente | ✗ → blind spot principale |
| Heading nello stato | assente | ✗ → POMDP aliasing |
| Maze M2 geometry | nuovo (dead-end rimossi) | ✓ |

---

## 2. Analisi run di oggi (soft update)

**Training (4000 ep, binary reward, 10 spawn):**

| Spawn | Avg steps | Max-steps% | Verdetto |
|---|---|---|---|
| F2 (-1.5,-4.0) | 402.9 | 0% | Dead-end — 3a conferma |
| F3 (6.0,6.0) | 338.2 | 28.8% | OK |
| F1 (-4.5,-3.5) | 333.9 | 40.0% | Migliore spawn |
| C2 (-7.0,5.0) | 304.7 | 1.9% | Dead-end — 3a conferma |
| A1 (-6.0,0.0) | 283.7 | 21.7% | OK |
| B3 (-4.5,1.5) | 228.7 | 0% | Sempre 0% (già rimosso in merge16_05) |
| E2 (0.0,3.5) | 140.5 | 0% | Problematico |
| D2 (0.5,-2.0) | 112.3 | 0% | Problematico |
| D1 (1.5,0.0) | 93.1 | 1.1% | Marginale |
| D3 (3.5,0.5) | 55.5 | 0% | Pessimo |

**Test M2 (90 ep, ε=0.0):** 40% global — paragonabile a merge16_05 ma con crash rate training più alto (90.6% vs 81%).

**Causa peggioramento:** soft update TAU=0.005 = target EMA time constant ≈ 200 step → più aggressivo di hard update 1000. Target mai veramente congelato → moving target problem → instabilità Q-learning (Mnih 2015).

---

## 3. Perché soft update è sbagliato per DQN

Soft update (Polyak averaging) nasce con **DDPG (Lillicrap et al. 2016)** per spazi d'azione **continui**. Non è mai stato proposto per DQN/DDQN.

- **Mnih et al. 2015 (DQN):** hard update ogni C step fornisce target stazionari per C step → supervisione stabile. Ha testato entrambi; hard update più stabile.
- **Van Hasselt et al. 2016 (DDQN):** mantiene hard update. Non propone soft update.
- **Con TAU=0.005:** target cambia ad ogni gradient step. Con operatore `max` discreto, ogni mini-batch insegue target diverso → deadly triad (Sutton & Barto: function approx + bootstrapping + off-policy).

**Conferma empirica:** crash rate 90.6% (soft) vs 80.8% (hard 5000). Instabilità aumentata.

**Fix:** TARGET_UPDATE_STEPS=5000 — già ripristinato nel commit f15e1a4. ✓

---

## 4. Perché PER è sbagliato per questo setup

`Analisi_softupdate_and_buffer.md` di Matteo raccomanda PER. Da non implementare.

**Feng 2021 §3.2:** PER testato direttamente sul paper di riferimento → risultati **peggiori** del campionamento uniforme. Motivazione: in collision avoidance, le transizioni con TD-error alto sono prevalentemente i crash → PER sovracampiona crash → policy diventa eccessivamente conservativa (si ferma prima degli ostacoli).

**Decisione irreversibile:** uniform replay è confermato corretto. Non toccare.

---

## 5. Il blind spot principale: assenza di contesto temporale

**Nessuno dei 10 training ha mai aggiunto frame stacking.** Questo è il problema fondamentale non affrontato.

Con LIDAR istantaneo (50 dim), il robot vede la distanza dalle pareti ma **non sa se si sta avvicinando o allontanando**. Stesso vettore LIDAR → stessa azione, indipendentemente dalla traiettoria. Questo causa:
- Wall-following loop (specialmente in M3)
- Dead-end exploitation (non capisce che sta oscillando)
- Seed brittleness amplificata (traiettorie memorizzate, non feature)

**Fix:** frame stacking k=3 → stato [scan_t, scan_t-1, scan_t-2] = 150 dim.

Hausknecht & Stone 2015 ("Deep Recurrent Q-Networks"): frame stacking k=4 risolve POMDP in molti task più efficacemente di LSTM. Implementazione: ~3 ore. Zero cambio architetturale.

---

## 6. Piano DDQN Enhanced — fix prioritizzate

### Fix OBBLIGATORIE (bug/regressioni)

**A. Min-pooling LIDAR** (`usv_logic.py`) — 30 min

```python
# REVERTIRE uniform sampling:
indices = np.linspace(0, len(scan) - 1, n_bins, dtype=int)
return scan[indices]

# RIPRISTINARE min-pooling:
chunks = np.array_split(scan, n_bins)
return np.array([np.min(chunk) for chunk in chunks])
```

**B. Rimuovere set_seed da train.py** — 10 min

```python
# RIMUOVERE da train.py (ogni blocco resetta RNG → bias sistematico):
# set_seed(42)
# Mantenere solo in test.py (ε=0.0 già deterministico)
```

**C. Rimuovere F2, C2, B3, D3 da SPAWN_LISTS[2]** (`usv_env.py`) — 20 min

Dead-end confermati su 3 run (F2, C2) e 0% su run odierna (B3, D3).
Spawn rimanenti: F1, F3, A1, D1, D2, E2 → **6 spawn** (riduzione rumore).

**D. Ri-validare spawn con maze nuovo** — 30 min
Maze M2 è cambiato geometricamente. Verificare clearance ≥ 0.40m per ogni spawn rimasto con test visivo (start_test_gui.sh).

### Fix AD ALTO IMPATTO (nuove feature)

**E. Frame stacking k=3** (`usv_env.py`, `ddqn_model.py`) — 3h

```python
# usv_env.py — buffer circolare
from collections import deque

class UsvEnv(Node):
    def __init__(self):
        ...
        self._frame_buffer = deque(maxlen=3)

    def get_state(self) -> np.ndarray:
        scan = (self.current_scan / LIDAR_MAX_RANGE).copy()
        self._frame_buffer.append(scan)
        # Padding: se buffer non pieno, replica primo frame
        while len(self._frame_buffer) < 3:
            self._frame_buffer.appendleft(scan)
        return np.concatenate(list(self._frame_buffer))  # 150 dim

    def reset_environment(self, ...):
        self._frame_buffer.clear()
        ...
```

```python
# ddqn_model.py — solo STATE_DIM
STATE_DIM = 150  # era 50
```

**F. Heading nello stato** (`usv_env.py`) — 1h

Richiede lettura yaw da `/gazebo/model_states` (già disponibile con plugin `gazebo_ros_state` aggiunto in f15e1a4).

```python
state = np.concatenate([stacked_scan, [math.cos(yaw), math.sin(yaw)]])
# 152 dim totali
```

**G. Multi-maze M2+M3** (`train.py`) — 2h

Pattern: 2 ep M2 → 1 ep M3 (ratio 2:1).
Cobbe et al. 2019: test set misura generalizzazione solo se training set include ambienti diversi.

```python
maze_id = 3 if (ep_global % 3 == 0) else 2
```

**H. space_bonus settore frontale** (`usv_logic.py`) — 30 min

```python
# Attuale (premia dead-end con vista aperta):
space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE

# Fix (solo beam frontali 15-35):
space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan[15:35])) / LIDAR_MAX_RANGE
```

### Fix DIFFERITE (fuori scope 2 settimane)

- Goal representation [dist, cos(angle), sin(angle)] — richiede definizione goal fisso per maze
- Continuous action space + SAC/TD3 — redesign completo
- LSTM policy — BPTT, complessità alta

---

## 7. Ordine di esecuzione raccomandato

```
Giorno 1:   A (min-pooling) + B (seed) + C (spawn) + D (validazione visiva)
Giorno 2:   E (frame stacking) — il più impattante
Giorno 3:   F (heading) + G (multi-maze)
Giorno 4:   Avvio training run1 (5000 ep)
Giorno 5-8: Training running
Giorno 9:   Analisi risultati, decide se serve run2
```

---

## 8. Success criteria

| Metrica | Baseline (merge16_05 best) | Target |
|---|---|---|
| M2 success | 46% | ≥ 46% (no regressione) |
| M3 success | 0% | > 0% (qualsiasi risultato) |
| Crash rate training | 80.8% | < 80% |
| Training stabile | avg100 oscilla ±250 | avg100 monotono dopo ep 3000 |
| Dead-end spawn | F2=0%, C2=0% | spawn rimossi → non misurato |

**Criterio go/no-go per run2:** se M3=0% dopo run1 con frame stacking → aggiungere heading e multi-maze per run2. Se M3>0% → ottimizzare iperparametri.

---

## 9. Ipotesi consolidate (NON modificare)

| Decisione | Motivazione |
|---|---|
| MSE loss + clip=10.0 | Huber+clip=1.0 → avg100<0 in `fixed_feng` |
| Uniform replay (no PER) | PER testato da Feng 2021 §3.2 → peggiora |
| GAMMA=0.99 | GAMMA=0.999 → orizzonte 1000 step > MAX_STEPS → Q esplodono |
| BETA_DECAY=0.999 | ε→0.05 a ep ~3000, orizzonte appropriato |
| REPLAY_START_SIZE=10,000 | Mnih 2015 — buffer diversificato prima del primo gradient step |
| TARGET_UPDATE=5000 | Van Hasselt 2016 — ~78 gradient steps per target shift |
| Hard update (no soft) | Soft update per spazi continui (DDPG), non DQN discreto |

---

## 10. Letteratura rilevante

- **Mnih et al. 2015** — DQN: hard update, uniform replay, frame stacking k=4 in Atari
- **Van Hasselt et al. 2016** — DDQN: target network stability, hard update ogni C step
- **Lillicrap et al. 2016** — DDPG: soft update *per spazi continui*, non applicabile qui
- **Hausknecht & Stone 2015** — DRQN: frame stacking k=4 risolve POMDP in molti task
- **Mirowski et al. 2016** — heading necessario per evitare aliasing in navigazione
- **Henderson et al. 2018** — seed brittleness, ≥5 seed per claim robusti
- **Ng et al. 1999** — potential-based shaping: invarianza policy solo se F=γΦ(s')-Φ(s)
- **Cobbe et al. 2019** — multi-environment training necessario per generalizzazione
- **Feng 2021** — paper di riferimento: MSE, uniform replay, BETA_DECAY=0.999

---

*Aggiornato: 2026-05-18 | Branch: matte_merge17_05*

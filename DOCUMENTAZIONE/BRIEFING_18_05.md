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

## 10. Track B — PPO parallelo (2 persone, opzionale)

### Motivazione

Con 4 persone nel gruppo, 2 persone possono condurre un esperimento parallelo con PPO mentre Track A fa DDQN enhanced. Obiettivo: confronto algoritmico per la presentazione finale.

**Perché PPO può battere DDQN su M3:**
- Policy stocastica a test time (distribuzione su azioni, non greedy) → no seed brittleness
- On-policy: ogni gradient step su dati freschi → meno overfitting su spawn specifici
- Shi et al. 2021: PPO supera DDQN su generalizzazione in indoor maze navigation
- Nessun ε-greedy da calibrare: entropia della policy gestisce esplorazione automaticamente

**Perché usare stable-baselines3 e non implementare da zero:**
- PPO già implementato, testato, documentato
- ~1 giorno per wrappare `usv_env.py` come `gym.Env`
- vs ~3-4 giorni per implementare PPO from scratch con rischio bug

### Architettura

**File nuovi (Track B non tocca nulla di Track A):**

```
src/my_usv/scripts/usv_gym_wrapper.py   ← gym.Env wrapper attorno a UsvEnv
src/my_usv/scripts/train_ppo.py         ← script training PPO
src/my_usv/scripts/test_ppo.py          ← script test (ε=0.0 equivalente)
```

**Dipendenza:** `pip install stable-baselines3` nel container Docker.

### Implementazione

**`usv_gym_wrapper.py`:**

```python
import gymnasium as gym
import numpy as np
from usv_env import UsvEnv, LIDAR_BEAMS
from usv_logic import LIDAR_MAX_RANGE

class UsvGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, maze_id: int = 2):
        super().__init__()
        self.env = UsvEnv()
        self.maze_id = maze_id
        # Stesso state space di Track A (con frame stacking se implementato)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(LIDAR_BEAMS,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(11)
        self._step_count = 0

    def reset(self, seed=None, options=None):
        obs = self.env.reset_environment(maze_id=self.maze_id)
        self._step_count = 0
        return obs, {}

    def step(self, action):
        obs, reward, done = self.env.step_action(int(action))
        self._step_count += 1
        truncated = (self._step_count >= 500)
        return obs, reward, done, truncated, {}
```

**`train_ppo.py`:**

```python
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from usv_gym_wrapper import UsvGymEnv
import rclpy

rclpy.init()
env = UsvGymEnv(maze_id=2)
check_env(env)  # verifica compatibilità gym

model = PPO(
    "MlpPolicy", env,
    verbose=1,
    n_steps=2048,          # raccolta esperienze per update
    batch_size=64,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    ent_coef=0.01,         # coefficiente entropia → esplorazione
    policy_kwargs=dict(net_arch=[256, 256])  # stessa dimensione di DDQN
)

model.learn(total_timesteps=2_500_000)  # ~5000 ep × 500 step
model.save("best_ppo_model")
rclpy.shutdown()
```

### Iperparametri PPO spiegati

| Param | Valore | Motivazione |
|---|---|---|
| `n_steps` | 2048 | Step raccolti prima di ogni update (on-policy rollout) |
| `n_epochs` | 10 | Passaggi sul rollout per ogni update (PPO clipping) |
| `ent_coef` | 0.01 | Peso entropia → esplorazione intrinseca, niente ε-greedy |
| `gamma` | 0.99 | Stesso di DDQN per confronto equo |
| `net_arch` | [256,256] | Stessa capacità di DDQN per confronto equo |
| `total_timesteps` | 2.5M | ~5000 ep × 500 step/ep |

### Coordinazione con Track A

- Track A e Track B usano **stesso `usv_env.py`** → nessun conflitto
- Se Track A ha già implementato frame stacking, Track B aggiorna `observation_space` di conseguenza
- Branch separati: Track A su `matte_merge17_05`, Track B su branch nuovo `ppo_merge17_05`
- Confronto finale: stessa metrica (success rate M2/M3, 90 ep, ε=0.0/deterministic)

### Success criteria Track B

| Metrica | Target |
|---|---|
| M2 success | Comparabile a Track A (±10%) |
| M3 success | > Track A → giustifica cambio algoritmo |
| Training convergenza | avg reward crescente entro 1M step |

### Timeline Track B

```
Giorno 1:  pip install SB3 nel container + test import
Giorno 2:  usv_gym_wrapper.py + check_env OK
Giorno 3:  train_ppo.py + avvio training
Giorno 4-8: Training running in parallelo a Track A
Giorno 9:  Confronto risultati
```

---

## 11. Letteratura rilevante

- **Mnih et al. 2015** — DQN: hard update, uniform replay, frame stacking k=4 in Atari
- **Van Hasselt et al. 2016** — DDQN: target network stability, hard update ogni C step
- **Lillicrap et al. 2016** — DDPG: soft update *per spazi continui*, non applicabile qui
- **Hausknecht & Stone 2015** — DRQN: frame stacking k=4 risolve POMDP in molti task
- **Mirowski et al. 2016** — heading necessario per evitare aliasing in navigazione
- **Henderson et al. 2018** — seed brittleness, ≥5 seed per claim robusti
- **Ng et al. 1999** — potential-based shaping: invarianza policy solo se F=γΦ(s')-Φ(s)
- **Cobbe et al. 2019** — multi-environment training necessario per generalizzazione
- **Schulman et al. 2017** — PPO: clipped objective, policy stocastica, on-policy stability
- **Shi et al. 2021** — PPO supera DQN variants su generalizzazione in indoor maze navigation
- **Feng 2021** — paper di riferimento: MSE, uniform replay, BETA_DECAY=0.999

---

---

## 12. Branch `ddqn_enhanced_18_05` — Log decisioni (2026-05-18)

### 12.1 Strategia di branching

**Decisione**: nuovo branch da `merge16_05`, NON fix-forward su `matte_merge17_05`.

**Motivazione**: analisi del diff tra i due branch ha rivelato che Matteo aveva regredito
tre componenti già validati:

| Componente | merge16_05 (corretto) | matte_merge17_05 (regredito) |
|---|---|---|
| LIDAR processing | min-pooling | uniform sampling (blind spot 5.3°) |
| Reward | shaping completo (~30 righe) | `return 5.0, False` binario |
| Spawn LISTS[2] | 6 punti, B3 rimosso | 10 punti, dead-end inclusi |

Ri-implementare reward shaping da zero = rischio typo su logica già testata su 3 run.
Portare solo le cose buone di Matteo (maze geometry + gazebo plugin) è più sicuro.

**Cosa è stato preso da `matte_merge17_05`**:
- `src/my_usv/worlds/Muri_9b/model.sdf`: pareti ridimensionate, dead-end geometrici rimossi, rename `Muri9b`
- `src/my_usv/worlds/labirinto_9b.world`: nuova geometria + plugin `gazebo_ros_state` (necessario per `/gazebo/set_entity_state`)
- Entrambi copiati via `git show matte_merge17_05:<path>` e committati

**Cosa è stato preservato da `merge16_05`**:
- Min-pooling in `usv_logic.py`
- Reward shaping completo (FRONT_DANGER, SIDE_DANGER, space_bonus, steering penalty)
- Spawn list M2 pulita (6 punti, B3 rimosso)
- TARGET_UPDATE_STEPS=5000

---

### 12.2 Fix E — Frame Stacking k=3

**Motivazione**: 10 training run senza mai provare contesto temporale. POMDP aliasing
strutturale: stesso vettore LIDAR in ingresso corridoio E in dead-end → stessi Q-values.
(Mirowski et al. 2016). Più alto impatto atteso per costo implementativo minimo.

**Implementazione**:
- `deque(maxlen=3)` in `UsvEnv.__init__`
- `_push_frame()` separato da `get_state()` — design critico (vedi §12.5)
- Padding episodio: `appendleft(scan_0)` × 2 → stato iniziale `[s0,s0,s0]` (Mnih 2015)
- `get_state()` = lettura pura, non modifica il buffer
- `STATE_DIM`: 50 → 150 (poi 152 con heading)

**Costante**:
```python
FRAME_STACK = 3    # Mnih et al. 2015
STEP_DT     = 0.1  # accoppiata con _wait_sim_seconds — unica source of truth
```

---

### 12.3 Fix F — Heading augmentation

**Prima idea (scartata)**: integrazione yaw da `cmd.angular_z * 0.1`.
**Problema**: USV in Gazebo ha inerzia idrodinamica, nessun tracking perfetto.
Drift accumulato su 500 step → heading "segnale comandato" non "segnale misurato".

**Scoperta**: `robot.urdf` ha `libgazebo_ros_planar_move.so` che pubblica `/odom` a 20 Hz.
Odom = pose esatta da Gazebo, zero drift, zero approssimazione.

**Implementazione finale**:
```python
self.odom_sub = self.create_subscription(Odometry, 'odom', self._odom_cb, 10)

def _odom_cb(self, msg: Odometry) -> None:
    q = msg.pose.pose.orientation
    self._current_yaw = math.atan2(
        2.0 * (q.w * q.z + q.x * q.y),
        1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    )
```

- Nessuna inizializzazione manuale yaw al reset: odom callback aggiorna `_current_yaw`
  durante `_wait_sim_seconds(0.3)` dentro `_teleport` e `_wait_sim_seconds(0.8)` nel reset
- Heading `[cos(yaw), sin(yaw)]` in `get_state()`: evita discontinuità 0/2π
- `STATE_DIM`: 150 → 152

---

### 12.4 Code review (subagent Opus)

Review approfondito ha trovato **2 Critical + 5 Important**. Tutti fixed.

**C1 — `get_state()` non idempotente (FIXED)**:
L'implementazione originale appendeva al frame buffer in `get_state()`.
Qualsiasi chiamata extra (debug, logger) avrebbe silenziosamente corrotto il buffer.

Fix: `_push_frame()` avanza il buffer, `get_state()` è pura lettura.
Chiamate esplicite a `_push_frame()` in `reset_environment()` e `step_action()`.

**C2 — `STATE_DIM=50` hardcoded nei test (FIXED)**:
`test_agent.py:6` e `test_replay_buffer.py:5` avevano `STATE_DIM = 50` hardcoded.
Con `STATE_DIM=152`, ogni test che chiama `agent.act(state)` avrebbe lanciato
shape mismatch su `Linear(152,300)` con input 50-dim.

Fix: `from ddqn_model import STATE_DIM` in entrambi i file.
`conftest.py` aggiunge `scripts/` a `sys.path` → import funziona. ✓

**I2 — yaw illimitato (FIXED)**:
Integrazione yaw cresceva senza bound. `% (2 * math.pi)` → ora normalizzato [0, 2π).
(Poi rimosso con l'odom subscriber, ma il pattern è documentato.)

**M1 — costante dt duplicata (FIXED)**:
`_wait_sim_seconds(0.1)` in `step_action` e `cmd.angular_z * 0.1` erano disaccoppiati.
`STEP_DT = 0.1` unica source of truth → entrambi ora usano `STEP_DT`.

**M5 — citazione errata (FIXED)**:
Commento citava Hausknecht & Stone 2015 (DRQN, recurrent) per frame stacking.
Frame stacking = Mnih et al. 2015 (DQN Atari, k=4). Corretto.

---

### 12.5 Test fixes

**test_usv_env.py** aveva 3 test scritti per spawn list a 16 punti (`2ff6530`):
- `test_maze2_spawn_count_is_16`: lista attuale ha 6 punti
- `test_maze2_includes_diagonal_yaw`: nessun yaw 45°/135° nella lista
- `test_maze2_spawn_covers_6_zones`: Zone C vuota con 6 punti

**Decisione**: aggiornare test per riflettere realtà, con TODO per re-validazione.
- `test_maze2_spawn_count_at_least_4`: >= 4, con commento "TODO: espandere su nuovo maze"
- `test_maze2_spawn_covers_minimum_zones`: solo A/D/F richieste (B/C/E da re-validare)
- `test_maze2_yaws_in_valid_range`: controlla [0, 2π), non yaw specifici

**nav_msgs stub mancante**: `_load_env_module()` non aveva `nav_msgs`/`nav_msgs.msg`
negli stub ROS. Aggiunto → import `from nav_msgs.msg import Odometry` funziona in test.

---

### 12.6 Bug install/ (scoperto durante primo training)

**Sintomo**: training bloccato su `Attendo /gazebo/set_entity_state...` per 20+ secondi.

**Causa**: `install/my_usv/share/my_usv/worlds/labirinto_9b.world` era la versione
pre-Matteo (senza plugin `gazebo_ros_state`). Il git checkout non modifica file untracked.
Il tentativo di copiare da git all'inizio della sessione era fallito con
"path exists on disk, but not in 'matte_merge17_05'" (install/ non tracciato da git).

**Fix**: copia manuale da `src/` a `install/`:
```bash
cp src/my_usv/worlds/labirinto_9b.world install/my_usv/share/my_usv/worlds/
cp src/my_usv/worlds/Muri_9b/model.sdf  install/my_usv/share/my_usv/worlds/Muri_9b/
```

**Regola appresa**: ogni volta che si cambia branch con modifiche a world files,
copiare manualmente in `install/` o lanciare `colcon build --packages-select my_usv`.

---

### 12.7 Stato finale branch

**Commit history**:
```
baf6354  feat(world): maze M2 aggiornato da matte_merge17_05
7f9b389  feat(state): frame stacking k=3 + heading (cos/sin) — 152 dim
b285cff  fix(state): get_state() idempotente + test STATE_DIM aggiornato
61bf2df  fix(test+env): test spawn aggiornati + guard buffer vuoto
fba3eef  fix(heading): yaw da odom subscriber reale invece di integrazione
```

**State vector**: 152 dim = 50 LIDAR × 3 frame + [cos(yaw), sin(yaw)]

**Limitazioni note**:
- Heading channel 2/152 dim: sotto Xavier init pesa 1.3% input.
  Monitorare `fc1.weight[:, 150:152].abs().mean()` vs `fc1.weight[:, :150].abs().mean()`
- Spawn list M2 (6 punti) non ri-validata sul nuovo maze geometry. Da fare prima del
  training lungo se gli spawn attuali mostrano problemi nelle prime ep.
- `odom` topic: verificato empiricamente (training avviato, nessun crash al primo ep)

**Target training**:
- 5000 ep, M2-only, `./start_train_multimaze.sh --reset`
- Success criteria: M2 ≥ 46% (pari a merge12_05 che non aveva frame stacking)
- Stretch goal: M2 ≥ 60%, M3 > 0%

---

## 13. Storia completa di tutti i branch

Analisi eseguita il 2026-05-18 con `git fetch --all`. Documentati in ordine cronologico di creazione.

---

### `main`
**Autore:** Davide Covolo  
**Stato:** base storica, non usato per training

Branch iniziale del progetto. Contiene le prime versioni di `usv_env.py`, `train.py`, `start_train.sh`. Spawn del robot fisso (M1: x=-3, y=-3). Step massimi portati a 1200. Fix al check container running. Nessun curriculum, nessun reward shaping, LIDAR processing basico. Serve solo come riferimento storico della struttura iniziale.

Commit chiave:
- `648ab1a` — spawn M1 cambiato a (-3, -3), file test rapido
- `59e4812` — MAX_STEPS 1000 → 1200
- `92d4562` — fix check stato container (existence vs running)

---

### `fast_sim`
**Autore:** Davide Covolo  
**Stato:** abbandonato, esperimento iniziale

Primo tentativo di ottimizzare la simulazione Gazebo per training headless (senza GUI). Modifiche a `patch_world.py` e `start_sim.sh` per controllare `real_time_update_rate`. Caricato anche un file tutorial DDQN. Il branch contiene documentazione operativa (`GUIDA OPERATIVA`) per lo startup giornaliero della simulazione.

Commit chiave:
- `31cc7ae` — modifiche Claude per simulazione headless, file `.sh` per avvio
- `89e49d8` — revisione `patch_world.py` (ridotto da 83 righe, refactoring), `start_sim.sh`
- `0508b5e` — README con note training e warning sulla velocità simulazione

**Motivo abbandono:** le ottimizzazioni sono state assorbite nel workflow principale (`start_training_curriculum.sh`). Branch mai integrato.

---

### `curriculum_learning`
**Autore:** Davide Covolo (+ Claude)  
**Stato:** completato, sostituito da `paper_implementation`

Introduzione del curriculum progressivo in due fasi:
- **Phase 1**: solo Maze 1 finché avg_reward(maze1, finestra 50 ep) > 1500
- **Phase 2**: 30% Maze 1 / 70% Maze 2 (probabilistico, controllato da `phase.txt`)

Aggiunte importanti:
- `start_training_curriculum.sh` riscritta con funzione `select_maze()` probabilistica
- `PHASE_FILE` (`phase.txt`) come file sentinel per transizione fase
- `MAX_STEPS` alzato da 500 → 1000
- Reward shaping: space bonus, quadratic front danger, FRONT_DANGER esteso a 3m

Fix incluso: `test_reward` falso positivo rimosso.

Commit chiave:
- `b5c6462` — fix test falso positivo space bonus
- `7bbbf7a` — feat reward: space bonus + quadratic front danger
- `8dc79c0` — MAX_STEPS 500 → 1000
- `26ebf0b` — phase detection + scrittura `phase.txt` quando maze1 avg50 > 1500
- `766a2c6` — select_maze() probabilistico, PHASE2_PROB=70

**Problema identificato post-hoc:** space bonus su mean(ALL scan) = dead-end attractor. Il robot impara a stare fermo al centro degli spazi aperti. Rimosso nelle iterazioni successive.

---

### `fixed_feng` (remote only, non fetchato localmente)
**Autore:** Matteo Bolo (BoloM03)  
**Stato:** FALLITO, non integrato

Tentativo di replicare più fedelmente Feng et al. 2021:
- **BATCH_SIZE**: 64 → **256**
- **Loss**: `MSELoss` → `SmoothL1Loss` (Huber Loss, per gestire reward spike -1000)
- **Gradient clipping**: 10.0 → **1.0** (comment: "evitare catastrophic forgetting")
- Aggiunto nuovo labirinto tra i world files (non integrato in train/test)
- Analisi parametri con confronto BATCH e EPISODE

Commit chiave (autore: BoloM03, 2026-05-09):
- `635112d` — file analisi parametri
- `98e1b5b` — BATCH 256 + Huber Loss + grad clip 1.0

**Diagnosi fallimento** (documentata in `feng_direct`):
- BATCH_SIZE=256 con buffer piccolo → overfitting early
- Huber Loss non aiuta se il problema è strutturale (dead-end aliasing)
- Non testato su Maze 2/3 → risultati non comparabili

---

### `feng_direct`
**Autore:** Davide Covolo  
**Stato:** analisi post-mortem, non training

Branch di analisi del fallimento `fixed_feng`. Contiene:
- Reorganizzazione repo: `analysis/`, `results/`, `models/`, `CHANGELOG`
- Documentazione strutturata in `DOCUMENTAZIONE/`
- Analisi `ANALISI_FIXED_FENG_FALLIMENTO` con diagnosi e riferimenti letteratura
- Spec training multimaze interleaved

Commit chiave:
- `2d56b74` — reorganizzazione repo
- `08b8049` — suite documentazione strutturata
- `a639323` — analisi fallimento fixed_feng con letteratura
- `2fda3f5` — spec training multimaze interleaved

**Nota:** da questo branch è emersa la decisione di non usare PER (Feng 2021 §3.2 lo aveva già testato con risultati peggiori).

---

### `gym_env`
**Autore:** Davide Covolo (+ Claude)  
**Stato:** prototipo funzionante, non usato in produzione

Wrapper `gymnasium` per compatibilità con algoritmi standard (PPO, SAC, ecc.):
- `usv_gym_env.py` — classe `UsvGymEnv(gym.Env)` con action space discreto e continuo
- `train_gym.py` — entry point training DDQN via gymnasium
- Test suite: 10 test failing → implementati → passing
- Docker: `gymnasium` aggiunto all'immagine

Commit chiave:
- `a4d456b` — test suite gymnasium (TDD: 10 test failing)
- `73815a5` — `UsvGymEnv`: observation/action space, reset/step, reward pass-through
- `f6c00d1` — `train_gym.py` DDQN via gymnasium
- `ba0b55d` — port modifiche `paper_implementation` in `gym_env`
- `6a65b3e` — port train.py changes + `GYM_ENV_GUIDE`

**Motivo non usato:** overhead del wrapper non giustificato per DDQN puro. Utile solo se si vuole testare PPO/SAC in futuro senza modificare il core.

---

### `paper_implementation`
**Autore:** Davide Covolo (+ Claude)  
**Stato:** completato, base per merge14_05

Replica il più fedelmente possibile Feng et al. 2021:
- **Reward semplificato**: solo +5 per step, -1000 collisione (rimossi tutti gli shaping)
- **Spawn per-episode**: `SetEntityState` teleport da `usv_env.py` (random da lista)
- **ε decay**: BETA_DECAY 0.995 → **0.999** (curva decay su 3000 ep intera)
- **Epsilon reset Phase 2**: `max(eps, 0.5)` quando Phase 2 si attiva (Narvekar et al. 2020)
- **gazebo_ros_state plugin**: aggiunto ai world file (primo branch a introdurlo)
- **maze_id** passato a `env.reset_environment()` per spawn selettivo

Commit chiave:
- `53e513a` — reward semplificato +5/-1000 (Feng et al.)
- `a8aa78b` — BETA_DECAY 0.995 → 0.999
- `f3278eb` — spawn per-episode via SetEntityState
- `5c1dd45` — epsilon reset max(eps, 0.5) a Phase 2
- `f86a478` — gazebo_ros_state plugin aggiunto a entrambi i world files

---

### `merge11_05`
**Autore:** Davide Covolo  
**Stato:** completato, primo test multimaze strutturato

Prima integrazione sistematica:
- `SPAWN_LISTS` e `TEST_SPAWN_LISTS` separati (train vs test riproducibili)
- Spawn M1 ridotto a 2 punti validati
- Spawn M2: fix B2→B3, rimossa F2
- Piano implementazione merge12_05 committato come spec

Commit chiave:
- `faf6628` — spawn M1 ridotto a 2 punti validati
- `7b55a40` — SPAWN_LISTS[3] + TEST_SPAWN_LISTS
- `1bfaf51` — fix TEST_SPAWN_LISTS M2
- `50866f4` — analisi risultati merge11_05 + spec merge12_05

---

### `merge12_05`
**Autore:** Davide Covolo  
**Stato:** COMPLETATO — baseline di riferimento

**Risultati**: M1=66.7%, M2=46.7%, M3=0%  
Primo training multimaze (M1+M2) con risultati soddisfacenti su M2. M3=0% atteso (mai visto in training). Spawn M2 ridotto da 16 → 10 punti (rimossi A2/B1/B2/C1/C3/E1 non sicuri).

Commit chiave:
- `9451a07` — M2 spawn 16 → 10 punti
- `d445451` — TOTAL_BLOCKS=45, BLOCK_SIZE=100 (4500 ep)
- `c1a2194` — risultati M1=66.7%, M2=46.7%, M3=0%
- `b3e41e9` — spec merge14_05 (REPLAY_START_SIZE + M2-only + spawn logging)

**Nota:** questo è il baseline che ddqn_enhanced_18_05 cerca di battere su M2.

---

### `merge14_05`
**Autore:** Davide Covolo  
**Stato:** COMPLETATO, risultati deludenti

M2-only training per isolare il maze difficile. run3: M2=20%, M3=13% zero-shot.
Introdotto `REPLAY_START_SIZE` (fill buffer prima di training). Spawn logging aggiunto.

---

### `merge15_05`
**Autore:** Davide Covolo  
**Stato:** COMPLETATO, overfitting posizionale

8000 ep, M2-only, 7 spawn points. Risultati: M2=13% — peggio di merge12_05.  
Diagnosi: overfitting posizionale (rete memorizza traiettorie specifiche invece di generalizzare). Evidenza: performance degrada man mano che si esaurisce la varianza degli spawn.

---

### `merge16_05`
**Autore:** Davide Covolo  
**Stato:** COMPLETATO — miglior baseline pre-frame-stack

Due run:
- **run1**: avg100=+197, M2=46% (pari a merge12_05 ma M2-only)
- **run2**: avg100=-65, M2=33% (instabilità)

Introduzioni chiave (non presenti in branch precedenti):
- Min-pooling (512 → 50 via `np.array_split` + min per bin)
- Reward shaping raffinato (no space bonus, front/side danger calibrati)
- TARGET_UPDATE=5000 (hard update, non soft)
- Spawn list M2 pulita (rimosso B3)

**Nota operativa**: questo è il branch da cui parte `ddqn_enhanced_18_05`.

---

### `matte_merge17_05`
**Autore:** Matteo Bolo (BoloM03) + Davide Covolo  
**Stato:** analizzato, NON usato come base

**Commit Matteo sul branch (2026-05-18)**:
- `3301eb1` — uniform sampling + set_seed(42) + TARGET_UPDATE=1000 (regressions)
- `7379e15` — soft update TAU=0.005 (wrong for DQN)
- `29002ff` — **revert a min-pooling** ("Ennesima cappata di Gemini tornati a minpool per il LIDAR")

Il commit `29002ff` è interessante: Matteo ha re-implementato min-pooling manualmente con loop esplicito (diverso dall'implementazione `np.array_split` di merge16_05, ma funzionalmente equivalente). Messaggio commit suggerisce che Gemini aveva suggerito uniform sampling e Matteo ha riconosciuto il problema tornando a min-pooling.

**Risultati training Matteo**: 4000 ep, crash rate 90.6%, M2=40% (peggio di merge16_05 run1=46%)

**Cosa è stato preso**: solo i world files (`labirinto_9b.world` + `Muri_9b/model.sdf`) con `gazebo_ros_state` plugin, cherry-picked in `ddqn_enhanced_18_05`.

---

### `ddqn_enhanced_18_05` ← BRANCH CORRENTE
**Autore:** Davide Covolo (+ Claude)  
**Stato:** IN TRAINING

Vedi §12 per dettaglio completo. Sintesi:
- Base: `merge16_05` (min-pooling, reward shaping, spawn puliti)
- Aggiunto da `matte_merge17_05`: world files M2 aggiornati (no dead-end, gazebo_ros_state)
- State: 152 dim = 50 LIDAR × 3 frame stack + [cos(yaw), sin(yaw)]
- Target: M2 ≥ 46%, stretch M2 ≥ 60%

---

*Aggiornato: 2026-05-18 | Branch: ddqn_enhanced_18_05*

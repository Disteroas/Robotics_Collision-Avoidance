# merge14_05 — Training Design Spec

**Data:** 2026-05-14  
**Branch:** `merge14_05` (da `merge12_05`)  
**Autore:** Davide Covolo

---

## Obiettivo

Verificare se il replay buffer prefilling (REPLAY_START_SIZE=10,000) migliora convergenza e generalizzazione rispetto alla baseline randomSpawn 05_08 (M2-only, M2=26.7%, M3=40%).

Variabile sperimentale principale: **REPLAY_START_SIZE** (Mnih 2015).  
Variabile secondaria: **M2-only training** (rimozione negative transfer M1→M3).  
Aggiunta diagnostica: **logging spawn point** per analisi cluster crash post-training.

---

## Motivazione

### 1. Replay buffer prefilling

Mnih et al. (2015) DQN: raccogliere 50,000 transizioni random prima del primo aggiornamento pesi. Standard non implementato in merge12_05.

**Bug attuale (`train_core.py:68`):**
```python
if len(self.memory) < BATCH_SIZE:   # 64 — troppo basso
    return None
```

Con BATCH_SIZE=64, training inizia dopo il primo episodio (86-216 step). I primi batch contengono 64 transizioni quasi identiche (stesso episodio, stessa traiettoria) → violazione dell'assunzione iid della experience replay (Lin 1992) → gradiente instabile.

Evidenza dal training log merge12_05:
```
ep1:  avg_loss = 110        ← 86 esperienze, altamente correlate
ep2:  avg_loss = 2,919      ← varianza enorme
ep3:  avg_loss = 5,849      ← rete si destabilizza
```

Con REPLAY_START_SIZE=10,000: buffer iniziale copre ~154 episodi M2 random (10 spawn, geometrie diverse) → primo batch è campione iid da distribuzione ampia.

### 2. M2-only training (rimozione negative transfer)

**Dato sperimentale:**

| Setup | Training | M3 (unseen) |
|-------|---------|-------------|
| randomSpawn 05_08 | M2-only, 3000 ep | **40%** |
| merge12_05 | M1+M2, 4500 ep | **0%** |

Aggiungere M1 azzera la generalizzazione a M3. Meccanismo: negative transfer (Pan & Yang 2010). M1 (geometria semplice, corridoi lunghi) introduce transizioni nel replay buffer che tirano i pesi verso comportamenti lineari. M3 (geometria complessa come M2) richiede comportamenti opposti ai chokepoint → crash.

M2 e M3 condividono caratteristiche geometriche locali (corridoi stretti, turns complessi). Policy da M2-only generalizza a M3 per similarità strutturale. Rimuovendo M1 si ripristina questo canale di generalizzazione.

### 3. Logging spawn point

Cluster crash osservati in merge12_05 test M2:
- Tipo A (5 ep): step=345, reward=+720 — spawn con chokepoint a metà percorso
- Tipo B (9 ep): step=113, reward=-440 — spawn con crash precoce

Senza logging dello spawn per episodio, impossibile identificare quale delle 10 posizioni causa quale cluster. Il logging permette analisi post-training con un semplice `awk`.

---

## Configurazione

### Parametri training

| Parametro | merge12_05 | merge14_05 |
|-----------|-----------|-----------|
| Maze training | M1+M2 (pattern 1:2) | **M2-only** |
| Episodi totali | 4500 | **4000** |
| Blocchi × ep/blocco | 45 × 100 | **20 × 200** |
| REPLAY_START_SIZE | 64 (BATCH_SIZE) | **10,000** |
| MAZE_PATTERN | (1 2 2) | **(2)** |
| MAX_STEPS | 500 | 500 |
| Spawn M2 training | 10 punti (A1,B3,C2,D1,D2,D3,E2,F1,F2,F3) | invariato |
| TEST_SPAWN_LISTS[2] | A1,B3,C2,D2,E2,F1 | invariato |
| best_avg in checkpoint | sì | invariato |

### Parametri invariati

GAMMA=0.99, LR=0.00025, BATCH_SIZE=64, MEMORY_CAPACITY=100,000, BETA_DECAY=0.999, TARGET_UPDATE_STEPS=1,000, reward +5/-1000, LIDAR processing, GAZEBO_SPEED=5×.

---

## Modifiche codice

### File 1: `train_core.py`

Aggiungere costante e modificare condizione `learn()`:

```python
# Accanto a BATCH_SIZE (riga ~27):
REPLAY_START_SIZE = 10_000

# In learn() (riga ~68):
def learn(self):
    if len(self.memory) < REPLAY_START_SIZE:   # era: BATCH_SIZE
        return None
    # resto invariato
```

### File 2: `usv_env.py`

Esporre ultimo spawn come attributo dell'ambiente:

```python
# In __init__:
self.last_spawn = (0.0, 0.0, 0.0)

# In reset_environment(), dopo il loop retry, prima di return:
self.last_spawn = (x, y, yaw)
self.accepting_scans = True
return self.get_state()
```

### File 3: `train.py`

Quattro cambiamenti:

**a. Leggere spawn dopo reset:**
```python
state = env.reset_environment(maze_id=args.maze_id)
sx, sy, syaw = env.last_spawn
spawn_label = f"({sx:.1f},{sy:.1f})"
```

**b. CSV header — aggiungere colonna `spawn`:**
```python
csv_w.writerow([
    'ep_global', 'maze', 'steps', 'reward',
    'avg100', 'epsilon', 'avg_loss', 'crashed',
    'total_steps', 'total_crashes', 'spawn'
])
```

**c. CSV row — aggiungere spawn_label:**
```python
csv_w.writerow([
    ep_disp, args.maze_id, steps + 1,
    round(ep_rew, 2), round(avg100, 2),
    round(agent.epsilon, 4), round(avg_loss, 6),
    int(done), agent.total_steps, crashes, spawn_label
])
```

**d. Terminal print — aggiungere spawn e log prefill:**
```python
# Variabile prima del loop:
_prefill_done = [False]

# Nel loop, dopo learn() (riga ~116):
if not _prefill_done[0] and len(agent.memory) >= REPLAY_START_SIZE:
    print(f"\n  ✅ PREFILL completato: {len(agent.memory)} transizioni. Training avviato.\n")
    _prefill_done[0] = True

# Nel print status (riga ~137-145): aggiungere spawn_label
print(
    f"Ep {ep_disp:4d}/{total_ep} [M{args.maze_id}] {status} | "
    f"sp:{spawn_label} | "
    f"R:{ep_rew:8.1f} | avg100:{avg100:8.1f} | "
    f"ε:{agent.epsilon:.3f} | loss:{avg_loss:.4f} | "
    f"crash:{crashes} [{bar}]"
)
```

**e. `--total-ep` default:** 4500 → 4000.

### File 4: `start_train_multimaze.sh`

```bash
TOTAL_BLOCKS=20      # era 45
BLOCK_SIZE=200       # era 100
MAZE_PATTERN=(2)     # era (1 2 2)
```

**Riga header commento (riga 5):**
```bash
# merge14_05 — 4000 episodi, 20 blocchi x 200 ep, M2-only, REPLAY_START_SIZE=10000
```

---

## Comportamento atteso

### Fase prefill (ep 1 → ~154)

- ε ≈ 1.0 → azioni quasi random → crash frequenti (~65 step avg su M2)
- Buffer cresce 0 → 10,000 (raggiunto a ep ~154: 10,000/65 ≈ 154 ep)
- `learn()` restituisce None → `losses=[]` → `avg_loss=0.0` loggato nel CSV
- Terminal: log normale per ogni ep + `"✅ PREFILL completato: 10,000 transizioni"` una volta
- Prefill completa entro block 1 (200 ep × 65 step avg = 13,000 step > 10,000) ✓

### Fase training (ep ~154 → 4000)

- Primo gradient update: buffer con 10,000 transizioni da tutti i 10 spawn M2
- ε → 0.05 a ep ~3000 (BETA_DECAY=0.999: `0.999^3000 ≈ 0.05`)
- Ultimi 1000 ep (3000-4000): ε=0.05, exploitation dominante

### Predizione risultati test

| Maze | Predizione | Baseline (randomSpawn 05_08) | merge12_05 |
|------|-----------|------------------------------|-----------|
| M1 | ~0-10% | 26.7% | 66.7% |
| M2 | ~50-65% | 26.7% | 46.7% |
| M3 | ~40-55% | 40.0% | 0% |

M3 ≥ 40%: conferma che negative transfer M1 era la causa di M3=0%.  
M3 > 40%: prefill contribuisce a migliore policy generalizzabile.  
M2 > 46.7%: prefill migliora convergenza vs M1+M2 training.

---

## Analisi post-training

Con colonna `spawn` nel CSV:

```bash
# Crash per spawn point:
awk -F, '$8==1{print $11}' training_log.csv | sort | uniq -c | sort -rn

# Success rate per spawn (test):
awk -F, '$5==0{print $11}' test_results.csv | sort | uniq -c
```

Identifica spawn problematici → decide quali rimuovere o verificare in Gazebo per merge15_05.

---

## Invarianti e decisioni confermate

- **NON cambiare:** MSE loss, clip=10.0, GAMMA=0.99, uniform replay (no PER)
- **NON aggiungere:** M3 in training (tenuto come test unseen per generalizzazione)
- **NON aggiungere:** [cos(yaw), sin(yaw)] — pianificato per merge15_05
- Spawn M2 training: 10 punti validati, invariati
- TEST_SPAWN_LISTS: 6 punti deterministici, invariati (confronto diretto con merge12_05)
- P2 M1: mantenuto in SPAWN_LISTS[1] ma irrilevante (M1 non in training)

---

## File modificati

| File | Tipo modifica | Righe cambiate |
|------|--------------|----------------|
| `train_core.py` | Add `REPLAY_START_SIZE`, fix `learn()` | 2 |
| `usv_env.py` | Add `last_spawn` attribute | 2 |
| `train.py` | Spawn logging, prefill notification, `--total-ep` default | ~10 |
| `start_train_multimaze.sh` | `TOTAL_BLOCKS`, `BLOCK_SIZE`, `MAZE_PATTERN`, header | 4 |

---

## Comando training

```bash
git checkout -b merge14_05
# ... implementazione ...
./start_train_multimaze.sh --reset
```

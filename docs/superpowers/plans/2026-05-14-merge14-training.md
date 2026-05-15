# merge14_05 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement REPLAY_START_SIZE=10,000 + M2-only training + spawn point logging su branch merge14_05.

**Architecture:** 4 file modificati, 3 modifiche funzionali indipendenti: (1) soglia prefill in train_core.py, (2) attributo last_spawn in usv_env.py, (3) logging spawn + prefill notification in train.py, (4) pattern M2-only in start_train_multimaze.sh. Ogni task è autonomo e committato separatamente.

**Tech Stack:** Python 3, PyTorch, ROS2/Gazebo (non richiesto per i test), pytest.

---

## File modificati

| File | Tipo | Cambiamento |
|------|------|-------------|
| `src/my_usv/scripts/train_core.py` | Modify | Add `REPLAY_START_SIZE=10_000`, fix `learn()` condition |
| `src/my_usv/scripts/usv_env.py` | Modify | Add `self.last_spawn` attribute |
| `src/my_usv/scripts/train.py` | Modify | Spawn logging in CSV + terminal, prefill notification, `--total-ep` default |
| `start_train_multimaze.sh` | Modify | `TOTAL_BLOCKS=20`, `BLOCK_SIZE=200`, `MAZE_PATTERN=(2)` |
| `src/my_usv/test/test_agent.py` | Modify | Add 2 test per `REPLAY_START_SIZE` threshold |

---

## Task 1: REPLAY_START_SIZE in train_core.py

**Files:**
- Modify: `src/my_usv/scripts/train_core.py:27,68`
- Test: `src/my_usv/test/test_agent.py`

**Contesto:** Attualmente `learn()` restituisce None se `len(buffer) < BATCH_SIZE` (64). Training inizia dopo 64 esperienze — violazione iid assumption. Fix: alzare soglia a 10,000.

- [ ] **Step 1: Crea branch merge14_05**

```bash
git checkout merge12_05
git checkout -b merge14_05
```

Expected: `Switched to a new branch 'merge14_05'`

- [ ] **Step 2: Scrivi i test che devono fallire**

Apri `src/my_usv/test/test_agent.py`. Aggiungi alla fine del file, dopo `test_learn_returns_none_when_buffer_empty`:

```python
from train_core import DDQNAgent, REPLAY_START_SIZE


def _fill_buffer(agent, n, reward=5.0):
    """Push n transitions into agent's replay buffer."""
    s = np.zeros(50, dtype=np.float32)
    for _ in range(n):
        agent.memory.push(s, 0, reward, s, False)


def test_learn_returns_none_below_replay_start_size():
    agent = DDQNAgent()
    _fill_buffer(agent, REPLAY_START_SIZE - 1)
    assert agent.learn() is None


def test_learn_returns_float_at_replay_start_size():
    agent = DDQNAgent()
    _fill_buffer(agent, REPLAY_START_SIZE)
    result = agent.learn()
    assert result is not None
    assert isinstance(result, float)
```

Nota: la seconda importazione `from train_core import ... REPLAY_START_SIZE` fallisce finché la costante non esiste — è l'errore atteso.

- [ ] **Step 3: Esegui i test per verificare che falliscano**

```bash
cd src/my_usv
python -m pytest test/test_agent.py::test_learn_returns_none_below_replay_start_size test/test_agent.py::test_learn_returns_float_at_replay_start_size -v
```

Expected: `ImportError: cannot import name 'REPLAY_START_SIZE' from 'train_core'`

- [ ] **Step 4: Implementa le modifiche in train_core.py**

Apri `src/my_usv/scripts/train_core.py`. Aggiungi la costante dopo `TARGET_UPDATE_STEPS` (riga ~27):

```python
TARGET_UPDATE_STEPS = 1_000
REPLAY_START_SIZE   = 10_000
```

Modifica `learn()` (riga ~68) — cambia solo la condizione:

```python
    def learn(self):
        if len(self.memory) < REPLAY_START_SIZE:
            return None
        s, a, r, s2, d = self.memory.sample(BATCH_SIZE)
```

Tutto il resto di `learn()` rimane invariato.

- [ ] **Step 5: Esegui i test per verificare che passino**

```bash
cd src/my_usv
python -m pytest test/test_agent.py::test_learn_returns_none_below_replay_start_size test/test_agent.py::test_learn_returns_float_at_replay_start_size -v
```

Expected:
```
test/test_agent.py::test_learn_returns_none_below_replay_start_size PASSED
test/test_agent.py::test_learn_returns_float_at_replay_start_size   PASSED
```

Nota: `test_learn_returns_none_below_replay_start_size` richiede pushare 9,999 transizioni — ~100ms, normale.

- [ ] **Step 6: Esegui tutta la suite test_agent.py**

```bash
cd src/my_usv
python -m pytest test/test_agent.py -v
```

Expected: tutti i test precedenti ancora PASS, 2 nuovi PASS. Se i test PyTorch falliscono per incompatibilità Windows (errore noto), è accettabile — eseguire dentro Docker se necessario.

- [ ] **Step 7: Commit**

```bash
git add src/my_usv/scripts/train_core.py src/my_usv/test/test_agent.py
git commit -m "feat(train): REPLAY_START_SIZE=10_000 — prefill buffer prima del training"
```

---

## Task 2: last_spawn in usv_env.py

**Files:**
- Modify: `src/my_usv/scripts/usv_env.py:86` (in `__init__`), `usv_env.py:166` (in `reset_environment`)
- Test: verifica manuale (UsvEnv non istanziabile in test senza ROS2 completo)

**Contesto:** `reset_environment()` sceglie spawn con `random.choice()` dentro un loop retry. L'attributo `last_spawn` deve essere impostato dopo il loop, con le coordinate effettivamente usate.

- [ ] **Step 1: Aggiungi `last_spawn` in `__init__`**

Apri `src/my_usv/scripts/usv_env.py`. In `UsvEnv.__init__`, dopo `self._lidar_checked = False` (riga ~87), aggiungi:

```python
        self._lidar_checked = False
        self.last_spawn     = (0.0, 0.0, 0.0)
```

- [ ] **Step 2: Imposta `last_spawn` in `reset_environment()`**

In `reset_environment()`, cerca la riga `self.accepting_scans = True` alla fine del metodo (riga ~166, dopo il loop retry). Aggiungi `self.last_spawn = (x, y, yaw)` immediatamente prima:

```python
        # Queste 2 righe vanno alla fine del metodo, prima di return
        self.last_spawn      = (x, y, yaw)
        self.accepting_scans = True
        return self.get_state()
```

Verifica che `(x, y, yaw)` siano le variabili del loop `for attempt in range(SPAWN_MAX_RETRIES)`. Il loop assegna `x, y, yaw = random.choice(spawn_list)` — dopo il loop, `x, y, yaw` contengono l'ultimo spawn tentato (quello che ha avuto successo o l'ultimo retry).

- [ ] **Step 3: Verifica sintattica**

```bash
python -c "import ast; ast.parse(open('src/my_usv/scripts/usv_env.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Verifica che gli import del modulo restino invariati**

```bash
cd src/my_usv
python -m pytest test/test_usv_env.py -v
```

Expected: tutti i test esistenti PASS (testano SPAWN_LISTS e TEST_SPAWN_LISTS — non la classe).

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/usv_env.py
git commit -m "feat(env): esponi last_spawn dopo reset_environment"
```

---

## Task 3: Spawn logging + prefill notification in train.py

**Files:**
- Modify: `src/my_usv/scripts/train.py`

**Contesto:** train.py chiama `env.reset_environment()` a inizio di ogni episodio. Dopo questo call, `env.last_spawn` contiene le coordinate usate. Dobbiamo: (a) leggere le coordinate, (b) aggiungerle al CSV, (c) stamparle nel terminale, (d) notificare quando il prefill è completato, (e) aggiornare `--total-ep` default.

- [ ] **Step 1: Aggiorna `--total-ep` default e docstring**

In `train.py`, riga ~9 nel docstring e riga ~51 nell'argparse:

```python
# docstring riga ~9:
#   --total-ep    INT   Episodi totali del training (default 4000, per progress bar)

# argparse riga ~51:
    p.add_argument('--total-ep',   type=int, default=4000)
```

- [ ] **Step 2: Aggiungi variabile prefill tracker prima del loop episodi**

In `main()`, dopo la chiamata a `load_ckpt` (riga ~67) e prima del loop `for offset in range(...)` (riga ~101), aggiungi:

```python
    _prefill_done = [False]
```

- [ ] **Step 3: Aggiorna CSV header**

Nel blocco che scrive il header CSV (riga ~79-84), aggiungi `'spawn'` come ultima colonna:

```python
    if is_new:
        csv_w.writerow([
            'ep_global', 'maze', 'steps', 'reward',
            'avg100', 'epsilon', 'avg_loss', 'crashed',
            'total_steps', 'total_crashes', 'spawn'
        ])
```

- [ ] **Step 4: Leggi spawn dopo reset e aggiungi a logica episodio**

All'inizio del loop episodio, subito dopo `env.reset_environment(...)` (riga ~105):

```python
        state  = env.reset_environment(maze_id=args.maze_id)
        sx, sy, _ = env.last_spawn
        spawn_label = f"({sx:.1f},{sy:.1f})"
        ep_rew = 0.0
        losses = []
```

- [ ] **Step 5: Aggiungi notifica prefill completato nel loop interno**

Nel loop interno `for steps in range(MAX_STEPS)`, dopo `agent.step_done()` (riga ~118):

```python
            agent.step_done()
            if not _prefill_done[0] and len(agent.memory) >= REPLAY_START_SIZE:
                print(f"\n  ✅ PREFILL completato: {len(agent.memory)} transizioni. Training avviato.\n")
                _prefill_done[0] = True
```

Nota: aggiungi `from train_core import ... REPLAY_START_SIZE` all'import esistente in riga ~36-39:

```python
from train_core import (
    DDQNAgent, save_ckpt, load_ckpt,
    EPSILON_MIN, BETA_DECAY, REPLAY_START_SIZE,
)
```

- [ ] **Step 6: Aggiorna il terminal print per includere spawn**

Sostituisci il blocco print esistente (riga ~137-145) con:

```python
        status = '💥 CRASH' if done else '✅ OK   '
        pct    = int(ep_disp / total_ep * 20)
        bar    = '█' * pct + '░' * (20 - pct)
        print(
            f"Ep {ep_disp:4d}/{total_ep} [M{args.maze_id}] {status} | "
            f"sp:{spawn_label} | "
            f"R:{ep_rew:8.1f} | avg100:{avg100:8.1f} | "
            f"ε:{agent.epsilon:.3f} | loss:{avg_loss:.4f} | "
            f"crash:{crashes} [{bar}]"
        )
```

- [ ] **Step 7: Aggiorna CSV row per includere spawn_label**

Sostituisci il `csv_w.writerow(...)` (riga ~147-153) con:

```python
        csv_w.writerow([
            ep_disp, args.maze_id, steps + 1,
            round(ep_rew, 2), round(avg100, 2),
            round(agent.epsilon, 4), round(avg_loss, 6),
            int(done), agent.total_steps, crashes, spawn_label
        ])
```

- [ ] **Step 8: Verifica sintattica**

```bash
python -c "import ast; ast.parse(open('src/my_usv/scripts/train.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "feat(train): spawn logging in CSV, prefill notification, total-ep=4000"
```

---

## Task 4: start_train_multimaze.sh — M2-only pattern

**Files:**
- Modify: `start_train_multimaze.sh`

**Contesto:** Cambiare da M1/M2/M2 interleaved (45×100 ep) a M2-only (20×200 ep). Tre righe cambiano: TOTAL_BLOCKS, BLOCK_SIZE, MAZE_PATTERN.

- [ ] **Step 1: Aggiorna header commento**

Riga 5 del file (commento `# merge12_05 ...`), sostituisci con:

```bash
# merge14_05 — 4000 episodi, 20 blocchi x 200 ep, M2-only, REPLAY_START_SIZE=10000
```

- [ ] **Step 2: Aggiorna le 3 variabili**

Trova e aggiorna (righe ~18-20):

```bash
TOTAL_BLOCKS=20      # era 45
BLOCK_SIZE=200       # era 100
MAZE_PATTERN=(2)     # era (1 2 2)
```

- [ ] **Step 3: Verifica con grep**

```bash
grep -E "TOTAL_BLOCKS|BLOCK_SIZE|MAZE_PATTERN" start_train_multimaze.sh
```

Expected:
```
TOTAL_BLOCKS=20
BLOCK_SIZE=200
MAZE_PATTERN=(2)
```

- [ ] **Step 4: Verifica calcolo TOTAL_EP**

```bash
grep "TOTAL_EP" start_train_multimaze.sh
```

Expected: `TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))` — deve calcolare 20×200=4000 automaticamente. Nessuna modifica necessaria.

- [ ] **Step 5: Commit**

```bash
git add start_train_multimaze.sh
git commit -m "feat(script): M2-only pattern, 20x200 ep, MAZE_PATTERN=(2)"
```

---

## Task 5: Verifica finale + push

- [ ] **Step 1: Verifica tutti i parametri chiave**

```bash
grep -n "REPLAY_START_SIZE\|MAX_STEPS\|TOTAL_BLOCKS\|BLOCK_SIZE\|MAZE_PATTERN\|last_spawn" \
  src/my_usv/scripts/train_core.py \
  src/my_usv/scripts/train.py \
  src/my_usv/scripts/usv_env.py \
  start_train_multimaze.sh
```

Expected (subset):
```
train_core.py:28:REPLAY_START_SIZE   = 10_000
train_core.py:69:        if len(self.memory) < REPLAY_START_SIZE:
train.py:41:MAX_STEPS = 500
train.py:51:    p.add_argument('--total-ep',   type=int, default=4000)
usv_env.py:88:        self.last_spawn     = (0.0, 0.0, 0.0)
usv_env.py:167:        self.last_spawn      = (x, y, yaw)
start_train_multimaze.sh:18:TOTAL_BLOCKS=20
start_train_multimaze.sh:19:BLOCK_SIZE=200
start_train_multimaze.sh:20:MAZE_PATTERN=(2)
```

- [ ] **Step 2: Esegui suite test completa**

```bash
cd src/my_usv
python -m pytest test/test_replay_buffer.py test/test_agent.py test/test_usv_env.py test/test_usv_logic.py -v
```

Expected: tutti i test esistenti PASS + 2 nuovi PASS (`test_learn_returns_none_below_replay_start_size`, `test_learn_returns_float_at_replay_start_size`). I test PyTorch che fallivano prima per incompatibilità Windows/Python 3.14 sono pre-esistenti e accettabili.

- [ ] **Step 3: Verifica git log**

```bash
git log --oneline -6
```

Expected (dal più recente):
```
feat(script): M2-only pattern, 20x200 ep, MAZE_PATTERN=(2)
feat(train): spawn logging in CSV, prefill notification, total-ep=4000
feat(env): esponi last_spawn dopo reset_environment
feat(train): REPLAY_START_SIZE=10_000 — prefill buffer prima del training
docs: spec merge14_05 — REPLAY_START_SIZE + M2-only + spawn logging
docs: merge12_05 risultati — M1=66.7%, M2=46.7%, M3=0%
```

- [ ] **Step 4: Push branch**

```bash
git push -u origin merge14_05
```

Expected: `Branch 'merge14_05' set up to track 'origin/merge14_05'`

---

## Comando avvio training

```bash
./start_train_multimaze.sh --reset
```

**Output atteso nei primi 200 ep:**
```
Ep    1/4000 [M2] 💥 CRASH | sp:(-6.0,0.0) | R:  -440.0 | avg100: -440.0 | ε:0.999 | loss:0.0000 | crash:1 [░░░░░░░░░░░░░░░░░░░░]
...
  ✅ PREFILL completato: 10000 transizioni. Training avviato.
...
Ep  155/4000 [M2] 💥 CRASH | sp:(-4.5,1.5) | R:  -440.0 | avg100: ... | ε:0.858 | loss:1234.5 | crash:... [░░░░░░░░░░░░░░░░░░░░]
```

**Analisi spawn post-training:**
```bash
# Crash per spawn point (training):
awk -F, 'NR>1 && $8==1 {print $11}' src/my_usv/scripts/training_log.csv | sort | uniq -c | sort -rn

# Success rate per spawn (test):
awk -F, 'NR>1 && $5==0 {print $11}' src/my_usv/scripts/test_results.csv | sort | uniq -c
```

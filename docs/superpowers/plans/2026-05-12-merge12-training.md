# merge12_05 — Fix MAX_STEPS + Blocchi + best_avg Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Applicare 3 fix a merge11_05 che hanno causato M2=0% nel test: MAX_STEPS=500, blocchi 100 ep, best_avg persistito nel checkpoint.

**Architecture:** 3 file toccati in ordine di dipendenza — train_core.py (API save/load) prima, poi train.py (usa la nuova API), poi start_train_multimaze.sh (parametri script). Nessun nuovo file. Nessuna modifica a reward, spawn, hyperparametri.

**Tech Stack:** Python 3, PyTorch, pickle, bash

---

## File toccati

| File | Tipo | Righe chiave |
|------|------|-------------|
| `src/my_usv/scripts/train_core.py` | Modifica | `save_ckpt` (L97), `load_ckpt` (L115) |
| `src/my_usv/scripts/train.py` | Modifica | L12, L41, L51, L67, L74, L90, L157 |
| `start_train_multimaze.sh` | Modifica | L8, L18, L19 |

---

### Task 1: Crea branch merge12_05

**Files:**
- Nessun file modificato — solo operazione git

- [ ] **Step 1: Crea branch da merge11_05**

```bash
git checkout merge11_05
git checkout -b merge12_05
```

Expected: `Switched to a new branch 'merge12_05'`

- [ ] **Step 2: Verifica di essere sul branch corretto**

```bash
git branch --show-current
```

Expected: `merge12_05`

---

### Task 2: Fix train_core.py — best_avg in save_ckpt e load_ckpt

Il bug: `best_avg` non viene salvato nel checkpoint. Ad ogni nuovo blocco train.py reimposta `best_avg = -float('inf')`, quindi il modello "best" salvato in blocco N può essere sovrascritto all'inizio del blocco N+1. Fix: aggiungere `best_avg` come parametro a `save_ckpt` e come valore di ritorno di `load_ckpt`.

**Files:**
- Modifica: `src/my_usv/scripts/train_core.py:97-132`

- [ ] **Step 1: Modifica save_ckpt — aggiungi parametro best_avg**

Riga corrente (L97):
```python
def save_ckpt(agent, episode, rh, crashes, path):
```

Sostituisci l'intera funzione `save_ckpt` (righe 97-112) con:

```python
def save_ckpt(agent, episode, rh, crashes, path, best_avg=-float('inf')):
    data = {
        'episode':        episode,
        'q_net':          agent.q_net.state_dict(),
        'target_net':     agent.target_net.state_dict(),
        'optimizer':      agent.optimizer.state_dict(),
        'epsilon':        agent.epsilon,
        'total_steps':    agent.total_steps,
        'replay_buffer':  list(agent.memory.buffer),
        'reward_history': list(rh),
        'crashes':        crashes,
        'best_avg':       best_avg,
    }
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    shutil.move(tmp, path)
```

- [ ] **Step 2: Modifica load_ckpt — leggi e restituisci best_avg**

Riga corrente (L115):
```python
def load_ckpt(agent, path, rh):
    if not os.path.exists(path):
        return 0, 0
```

Sostituisci l'intera funzione `load_ckpt` (righe 115-132) con:

```python
def load_ckpt(agent, path, rh):
    if not os.path.exists(path):
        return 0, 0, -float('inf')
    print(f"  📂 Checkpoint: {path}")
    with open(path, 'rb') as f:
        d = pickle.load(f)
    agent.q_net.load_state_dict(d['q_net'])
    agent.target_net.load_state_dict(d['target_net'])
    agent.optimizer.load_state_dict(d['optimizer'])
    agent.epsilon       = d['epsilon']
    agent.total_steps   = d['total_steps']
    agent.memory.buffer = deque(d['replay_buffer'], maxlen=MEMORY_CAPACITY)
    rh.extend(d.get('reward_history', []))
    ep        = d['episode']
    crashes   = d.get('crashes', 0)
    best_avg  = d.get('best_avg', -float('inf'))
    print(f"  ↳ Ep:{ep} | ε:{agent.epsilon:.3f} | "
          f"Buffer:{len(agent.memory.buffer)} | Crash:{crashes} | "
          f"best_avg:{best_avg:.1f}")
    return ep, crashes, best_avg
```

- [ ] **Step 3: Verifica sintattica del file**

```bash
python3 -c "import ast, sys; ast.parse(open('src/my_usv/scripts/train_core.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Verifica API save_ckpt accetta best_avg**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src/my_usv/scripts')
import inspect, train_core
sig_save = inspect.signature(train_core.save_ckpt)
sig_load = inspect.signature(train_core.load_ckpt)
assert 'best_avg' in sig_save.parameters, 'save_ckpt manca best_avg'
print('save_ckpt:', sig_save)
print('load_ckpt:', sig_load)
print('OK')
"
```

Expected output (example):
```
save_ckpt: (agent, episode, rh, crashes, path, best_avg=-inf)
load_ckpt: (agent, path, rh)
OK
```

- [ ] **Step 5: Commit**

```bash
git add src/my_usv/scripts/train_core.py
git commit -m "fix(checkpoint): best_avg persiste in save_ckpt/load_ckpt"
```

---

### Task 3: Fix train.py — MAX_STEPS + docstring + chiamate a load_ckpt/save_ckpt

Tre sotto-fix nello stesso file:
1. `MAX_STEPS = 500` (era 1000) — causa root di M2=0%
2. Aggiornamento docstring e `--total-ep` default a 4500
3. Chiamate a `load_ckpt` e `save_ckpt` aggiornate per usare `best_avg`

**Files:**
- Modifica: `src/my_usv/scripts/train.py`

- [ ] **Step 1: Fix MAX_STEPS (riga 41)**

Trova:
```python
MAX_STEPS = 1000
```

Sostituisci con:
```python
MAX_STEPS = 500
```

- [ ] **Step 2: Aggiorna docstring — righe 11-14**

Trova:
```python
Calibrazione epsilon:
  Con BETA_DECAY=0.999 e 5000 episodi:
    ε dopo 1000 ep = 0.999^1000 = 0.368
    ε dopo 3000 ep = 0.050               → minimo raggiunto
```

Sostituisci con:
```python
Calibrazione epsilon:
  Con BETA_DECAY=0.999 e 4500 episodi:
    ε dopo 1000 ep = 0.999^1000 = 0.368
    ε dopo 3000 ep = 0.050               → minimo raggiunto
```

- [ ] **Step 3: Aggiorna default --total-ep (riga 51)**

Trova:
```python
    p.add_argument('--total-ep',   type=int, default=5000)
```

Sostituisci con:
```python
    p.add_argument('--total-ep',   type=int, default=4500)
```

- [ ] **Step 4: Aggiorna chiamata a load_ckpt (riga 67) e rimuovi best_avg = -inf (riga 74)**

Trova (righe 67-75):
```python
    last_ep, crashes = load_ckpt(agent, args.checkpoint, rh)

    if last_ep >= args.end_ep:
        print(f"  Blocco {args.start_ep}-{args.end_ep} già completato.")
        env.destroy_node(); rclpy.shutdown(); return

    ep_start = max(last_ep, args.start_ep)
    best_avg = -float('inf')
    total_ep = args.total_ep
```

Sostituisci con:
```python
    last_ep, crashes, best_avg = load_ckpt(agent, args.checkpoint, rh)

    if last_ep >= args.end_ep:
        print(f"  Blocco {args.start_ep}-{args.end_ep} già completato.")
        env.destroy_node(); rclpy.shutdown(); return

    ep_start = max(last_ep, args.start_ep)
    total_ep = args.total_ep
```

- [ ] **Step 5: Aggiorna save_ckpt nel signal handler (riga 90)**

Trova:
```python
        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint)
```

Sostituisci con:
```python
        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint, best_avg)
```

- [ ] **Step 6: Aggiorna save_ckpt periodico (riga 157)**

Trova:
```python
            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint)
```

Sostituisci con:
```python
            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint, best_avg)
```

- [ ] **Step 7: Verifica sintattica del file**

```bash
python3 -c "import ast; ast.parse(open('src/my_usv/scripts/train.py').read()); print('OK')"
```

Expected: `OK`

- [ ] **Step 8: Verifica valori critici**

```bash
grep -n "MAX_STEPS\|default=4500\|best_avg" src/my_usv/scripts/train.py
```

Expected (ordine e righe possono variare):
```
41:MAX_STEPS = 500
51:    p.add_argument('--total-ep',   type=int, default=4500)
67:    last_ep, crashes, best_avg = load_ckpt(agent, args.checkpoint, rh)
90:        save_ckpt(agent, _ep[0], rh, _cr[0], args.checkpoint, best_avg)
157:            save_ckpt(agent, ep_disp, rh, crashes, args.checkpoint, best_avg)
```

Verifica che `best_avg = -float('inf')` NON appaia più:
```bash
grep "best_avg = -float" src/my_usv/scripts/train.py
```

Expected: nessun output (riga rimossa)

- [ ] **Step 9: Commit**

```bash
git add src/my_usv/scripts/train.py
git commit -m "fix(train): MAX_STEPS 1000→500, best_avg da checkpoint, default 4500 ep"
```

---

### Task 4: Fix start_train_multimaze.sh — TOTAL_BLOCKS + BLOCK_SIZE + header

**Files:**
- Modifica: `start_train_multimaze.sh:8,18,19`

- [ ] **Step 1: Aggiorna header comment (riga 8)**

Trova:
```bash
#  5000 episodi, 25 blocchi x 200 ep, pattern (M1, M2, M2) ciclico.
```

Sostituisci con:
```bash
#  4500 episodi, 45 blocchi x 100 ep, pattern (M1, M2, M2) ciclico.
```

- [ ] **Step 2: Aggiorna TOTAL_BLOCKS (riga 18)**

Trova:
```bash
TOTAL_BLOCKS=25
```

Sostituisci con:
```bash
TOTAL_BLOCKS=45
```

- [ ] **Step 3: Aggiorna BLOCK_SIZE (riga 19)**

Trova:
```bash
BLOCK_SIZE=200
```

Sostituisci con:
```bash
BLOCK_SIZE=100
```

- [ ] **Step 4: Verifica sintattica dello script**

```bash
bash -n start_train_multimaze.sh && echo "Syntax OK"
```

Expected: `Syntax OK`

- [ ] **Step 5: Verifica valori**

```bash
grep -n "TOTAL_BLOCKS\|BLOCK_SIZE\|episodi" start_train_multimaze.sh | head -10
```

Expected:
```
8:#  4500 episodi, 45 blocchi x 100 ep, pattern (M1, M2, M2) ciclico.
18:TOTAL_BLOCKS=45
19:BLOCK_SIZE=100
```

- [ ] **Step 6: Verifica calcolo TOTAL_EP**

`TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))` è calcolato dinamicamente nello script — non richede modifica. Verifica:

```bash
bash -c "TOTAL_BLOCKS=45; BLOCK_SIZE=100; TOTAL_EP=\$(( TOTAL_BLOCKS * BLOCK_SIZE )); echo TOTAL_EP=\$TOTAL_EP"
```

Expected: `TOTAL_EP=4500`

- [ ] **Step 7: Commit**

```bash
git add start_train_multimaze.sh
git commit -m "fix(script): TOTAL_BLOCKS=45, BLOCK_SIZE=100 — 4500 ep, blocchi 100"
```

---

### Task 5: Verifica finale end-to-end

- [ ] **Step 1: Verifica tutti e 3 i fix insieme**

```bash
echo "=== MAX_STEPS ===" && grep "MAX_STEPS" src/my_usv/scripts/train.py
echo "=== BLOCKS ===" && grep "TOTAL_BLOCKS\|BLOCK_SIZE" start_train_multimaze.sh
echo "=== best_avg in save_ckpt ===" && grep "best_avg" src/my_usv/scripts/train_core.py
```

Expected:
```
=== MAX_STEPS ===
MAX_STEPS = 500
=== BLOCKS ===
TOTAL_BLOCKS=45
BLOCK_SIZE=100
=== best_avg in save_ckpt ===
def save_ckpt(agent, episode, rh, crashes, path, best_avg=-float('inf')):
        'best_avg':       best_avg,
    best_avg  = d.get('best_avg', -float('inf'))
    return ep, crashes, best_avg
```

- [ ] **Step 2: Verifica compatibilità backward del checkpoint**

Un checkpoint vecchio (senza `best_avg`) deve caricarsi correttamente. Il `.get('best_avg', -float('inf'))` lo garantisce. Verifica:

```bash
python3 -c "
import sys; sys.path.insert(0, 'src/my_usv/scripts')
import pickle, os
# Simula checkpoint vecchio (senza best_avg)
dummy = {'episode': 42, 'crashes': 5, 'reward_history': [], 'replay_buffer': [],
         'epsilon': 0.5, 'total_steps': 1000}
# Verifica che get con default funzioni
best_avg = dummy.get('best_avg', -float('inf'))
assert best_avg == -float('inf'), f'Expected -inf, got {best_avg}'
print('Backward compat OK: best_avg defaults to -inf on old checkpoint')
"
```

Expected: `Backward compat OK: best_avg defaults to -inf on old checkpoint`

- [ ] **Step 3: Verifica git log del branch**

```bash
git log --oneline -5
```

Expected (3 commit nuovi + i precedenti di merge11_05):
```
<sha> fix(script): TOTAL_BLOCKS=45, BLOCK_SIZE=100 — 4500 ep, blocchi 100
<sha> fix(train): MAX_STEPS 1000→500, best_avg da checkpoint, default 4500 ep
<sha> fix(checkpoint): best_avg persiste in save_ckpt/load_ckpt
...commit merge11_05...
```

- [ ] **Step 4: Avviare training**

```bash
./start_train_multimaze.sh --reset
```

Il banner di avvio deve mostrare:
```
  Episodi tot  : 4500
  Blocchi      : 45 x 100 ep
```

Il primo episodio M1 deve terminare in ≤ 500 step (non 1000).

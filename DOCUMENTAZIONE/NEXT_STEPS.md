# Next Steps — backlog tecnico prioritizzato

Basato su analisi progressiva da `feng_direct` → `merge14_05`.  
**Aggiornato 2026-05-14** — `merge14_05` pianificato: REPLAY_START_SIZE + M2-only + spawn logging. Da implementare.  
Ordinato per impatto atteso.

---

## Priorità ALTA — cambiano la struttura del task

### 0. Analizzare il gap con i risultati di Feng 2021 [NUOVO — fare prima di tutto]

**Perché:** `feng_direct` (fedele al paper) ottiene 10% successi su Maze 2 test. Feng ottiene 0 collisioni in 5 minuti. Il gap non è spiegato dai hyperparametri.

**Azioni:**
1. Rieseguire test con metrica "5 minuti continuativi" invece di episodi discreti da spawn fissi
2. Confrontare difficoltà Maze 2 con Map 2 di Feng (passaggi stretti, area totale)
3. Verificare se il modello `best_ddqn_model.pth` fa effettivamente collision avoidance o solo sopravvive in zone aperte

**File:** `test.py` (modificare metrica), `start_test_gui.sh` (osservare traiettoria)

---

### 1. Aggiungere goal allo stato ⚠️ ESTENSIONE — oltre scope Feng 2021

**Nota (2026-05-10):** Feng 2021 NON include goal nello stato (Eq.1: `st = Ot`, solo LIDAR). Goal = lavoro futuro (§6). Questo è un'estensione, non un fix del paper.

**Perché comunque utile:** senza goal l'agente impara collision avoidance pura. Per navigazione direzionale serve la destinazione nello stato.

**Come:** Definire un goal fisso per ogni maze (es. `GOAL_POSITIONS = {1: (x,y), 2: (x,y), 3: (x,y)}`). Aggiungere al vettore stato:
```python
dx = goal_x - robot_x
dy = goal_y - robot_y
dist = sqrt(dx**2 + dy**2) / MAX_DIST          # normalizzato [0,1]
angle_to_goal = atan2(dy, dx) - robot_yaw
state = np.concatenate([lidar_50bins, [dist, cos(angle_to_goal), sin(angle_to_goal)]])
# 50 → 53 dimensioni
```
Richiede posizione robot da Gazebo (topic `/gazebo/model_states` o TF).

**Effetto atteso:** riduzione crash rate da 90% a ~30-50% su Maze 2 in 3000 episodi (letteratura: Zhu et al. 2017, Mirowski et al. 2016).

**File da modificare:** `usv_env.py` (get_state, step_action, reset), `ddqn_model.py` (STATE_DIM: 50→53).

---

### 2. Reward shaping graduato

**Perché:** +5/-1000 non dà segnale prima della collisione. Nessun gradiente di pericolo.

**Come:**
```python
# In usv_logic.py, compute_reward():
if min_lidar < COLLISION_DIST:        # < 0.25m
    return -1000.0, True

danger = max(0.0, 1.0 - min_lidar / 1.0)   # 0 se libero, 1 se a 0m
reward = +5.0 - 30.0 * danger**2            # max = +5, min ≈ -25
return reward, False
```
Aggiungere bonus raggiungimento goal: `+200` quando `dist_to_goal < 0.5m` (fine episodio positiva).

**Note:** la forma `f(s) - f(s')` preserva la policy ottimale (Ng et al. 1999). Il reward di pericolo basato su min_lidar è una funzione potenziale valida.

**File da modificare:** `usv_logic.py` (compute_reward).

---

### 3. Multi-maze training ✅ COMPLETATO su `merge11_05` — risultati: M1=57%, M2=0%, M3=0%

**Perché:** Training su singolo maze → 0% generalizzazione su Maze 1 e 3. Cobbe et al. (2019) dimostrano che servono ambienti multipli.

**Risultato (merge11_05):** M1=57% ✓, M2=0% ✗, M3=0% ✗. Causa M2=0%: MAX_STEPS=1000 in training (bug).

**Fix → branch `merge12_05`:**
```bash
# da merge11_05, creare merge12_05:
git checkout -b merge12_05
# fix train.py MAX_STEPS=500, start_train_multimaze.sh TOTAL_BLOCKS=45 BLOCK_SIZE=100
# fix train_core.py best_avg in checkpoint
./start_train_multimaze.sh --reset
```

### 3b. Fix MAX_STEPS ✅ IDENTIFICATO — implementare in `merge12_05` [PRIORITÀ ALTA]

**Ora:** training=1000, test=500 → task distribution diversa → M2 policy non convergita.

**Fix (`train.py:41`):**
```python
MAX_STEPS = 500   # era 1000
```

**Motivazione (Tobin 2017):** con 16 spawn random, 500 step/episodio copre la distribuzione. GAMMA=0.99 → orizzonte effettivo = 100 step → 500 step = 5 orizzonti per episodio (ampiamente sufficiente).

### 3c. Fix best_avg checkpoint — implementare in `merge12_05` [PRIORITÀ MEDIA]

**Ora:** `best_avg = -float('inf')` in `train.py` si resetta ad ogni blocco. Il modello "best" può essere salvato con policy subottimale durante la transizione tra blocchi.

**Fix (`train_core.py`):**
```python
# In save_ckpt: aggiungere 'best_avg': best_avg
# In load_ckpt: return ep, crashes, d.get('best_avg', -float('inf'))
```

---

## Priorità MEDIA — migliorano stabilità training

### 4. ~~Huber loss + gradient clip~~ ❌ RIMOSSO — testato e fallito

**Aggiornamento 2026-05-10:** `fixed_feng` ha implementato esattamente questo fix. Risultato: avg100 < 0 dopo 3000 ep (peggiorato rispetto a `feng_direct` che raggiungeva +391 con MSE).

**Perché fallisce:** Feng 2021 usa MSE pura (Eq.5), nessun grad_clip. Huber(δ=1) + clip=1.0 riduce il segnale di apprendimento dai crash di ~10.000×. La combinazione clip=1.0 è valida solo se abbinata a reward clipping [-1,+1] (come in Mnih 2015 DQN originale).

**Decisione:** mantenere MSE + clip=10.0 (configurazione `feng_direct` funzionante). Vedere [ANALISI_FIXED_FENG_FALLIMENTO.md](ANALISI_FIXED_FENG_FALLIMENTO.md).

---

### 5. Contesto temporale nello stato

**Perché:** LIDAR snapshot è un POMDP. Heading e velocità angolare non sono osservati → la rete non può distinguere "mi sto avvicinando al muro" da "mi sto allontanando".

**Opzione A (semplice):** aggiungere `[cos(yaw), sin(yaw), angular_vel]` allo stato (50 → 53 dim). Richiede heading da Gazebo.

**Opzione B (più potente):** stack degli ultimi 3 frame LIDAR (50×3 = 150 dim). Come DQN su Atari.

**Opzione C (più potente ma complessa):** sostituire MLP con DRQN (LSTM). Richiede refactor del training loop.

**Raccomandazione:** iniziare con Opzione A (semplice, leggibile).

---

### 6. ~~Prioritized Experience Replay (PER)~~ ❌ RIMOSSO — scartato dal paper originale

**Aggiornamento 2026-05-10:** Feng 2021 (§3.2, p.6) ha testato esplicitamente DDQN+PER:
> *"the reward of DDQN with PER converged faster than the original DDQN but achieved a lower value in the end. Therefore, our obstacle avoidance method was developed based on DDQN."*

**Perché PER peggiora:** con reward +5/−1000, PER campiona crash con priorità ~200× superiore a survival. La rete diventa iper-conservativa e rifiuta i corridoi stretti dove deve necessariamente avvicinarsi alle pareti. Reward finale inferiore.

**Decisione:** non implementare PER. Uniform sampling rimane la scelta corretta per questo task.

---

## Priorità BASSA — fix e cleanup

### 7. Uniformare MAX_STEPS training/test

**Ora:** training = 1000, test = 500. Commento in `test.py` dice "coerente" ma è sbagliato.

**Fix:** scegliere un valore (500 è più ragionevole — con γ=0.99 l'orizzonte effettivo è ~100 step) e usarlo ovunque. Aggiornare `train.py` e `test.py`.

---

### 8. Re-test con bug spawn corretto

**Ora:** bug `test.py` (maze_id mancante) fixato nel commit `4bbc476`. Il test di `feng_direct` già eseguito aveva il bug attivo su Maze 2 e 3.

**Azione:** rieseguire `./start_test.sh` con il fix per avere dati puliti su Maze 2 (attesi ~80-85% crash, non 90%, eliminando i 7 crash step=1 da spawn errato).

---

## Roadmap sintetica (aggiornata 2026-05-14)

```
COMPLETATO:
  merge12_05: MAX_STEPS=500 + 100-ep blocks + best_avg fix + spawn reduction
  → M1=66.7% ✓, M2=46.7% ✓, M3=0% ✗
  → M2 risolto. M3=0%: negative transfer da M1 training (causa identificata)

ITERAZIONE CORRENTE (branch merge14_05 — pianificato):
  REPLAY_START_SIZE=10,000 + M2-only training + spawn logging
  - Prefill: Mnih 2015 — buffer diversificato prima del primo gradient step
  - M2-only: rimozione negative transfer M1→M3 (Pan & Yang 2010)
  - Spawn logging: diagnostica cluster crash per spawn point
  → target: M2 ≥50%, M3 ≥40% (ripristino baseline randomSpawn 05_08 + miglioramento)
  4000 ep, 20 blocchi × 200 ep

ITERAZIONE SUCCESSIVA (merge15_05 — dopo risultati merge14_05):
  [cos(yaw), sin(yaw)] → 50→52 dim stato
  - Risolve state aliasing (causa P2 crash in M1, generalizzazione limitata)
  - Mirowski et al. 2016: heading nello stato riduce POMDP aliasing
  - Prerequisito: training from scratch (architettura cambia)
  - Opzione: reintrodurre M1 con ratio ridotto (1:4 o 1:5) per mantenere M1 senza negative transfer

ITERAZIONE B (se merge15_05 non basta):
  Frame stacking (ultimi 3 LIDAR → 150 dim)
  - Mnih 2015 Atari: cattura direzione di movimento implicita
  - Alternativa più semplice a DRQN (Hausknecht & Stone 2015)

ITERAZIONE C (estensioni, opzionale):
  fix 1 (goal in stato) → va oltre scope Feng 2021, solo se B è stabile
  fix 2 (reward shaping graduato) → solo se architettura base è solida
  NON fare: Huber, clip=1.0, PER — tutti testati e peggiorano
```

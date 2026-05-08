# Next Steps — backlog tecnico prioritizzato

Basato su analisi `feng_direct` ([report_feng_direct.md](report_feng_direct.md)).  
Ordinato per impatto atteso. Le prime 3 voci sono prerequisiti per qualsiasi miglioramento significativo.

---

## Priorità ALTA — cambiano la struttura del task

### 1. Aggiungere goal allo stato

**Perché:** Lo stato attuale è solo LIDAR. L'agente non ha destinazione → impara collision avoidance pura, non navigazione. Feng 2021 include distanza e angolo al goal nello stato.

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

### 3. Multi-maze training

**Perché:** Training su singolo maze → 0% generalizzazione su Maze 1 e 3. Cobbe et al. (2019) dimostrano che servono ambienti multipli.

**Come:** Alternare Maze 1 e Maze 2 ogni 200-300 episodi (stessa logica di `paper_implementation` ma con epsilon e threshold corretti).

**Attenzione:**
- Usare `BETA_DECAY=0.999` (non 0.995)
- Non resettare epsilon al cambio maze
- Non usare threshold di phase transition basata su avg reward — usare success rate >= 20% su finestra 100 ep

**File da modificare:** `start_train_direct.sh`, `train.py`.

---

## Priorità MEDIA — migliorano stabilità training

### 4. Huber loss + gradient clip

**Perché:** MSE con reward bimodale (+5/-1000) produce gradienti instabili. avg_loss a fine training = 2000-3000 (non convergita).

**Come (2 righe in `train_core.py`):**
```python
self.loss_fn = nn.SmoothL1Loss()          # era: nn.MSELoss()
# ...
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)   # era: 10.0
```

**Nota:** cambia il comportamento del training, il checkpoint esistente diventa parzialmente incompatibile (si può caricare ma il training riprende con loss diversa).

---

### 5. Contesto temporale nello stato

**Perché:** LIDAR snapshot è un POMDP. Heading e velocità angolare non sono osservati → la rete non può distinguere "mi sto avvicinando al muro" da "mi sto allontanando".

**Opzione A (semplice):** aggiungere `[cos(yaw), sin(yaw), angular_vel]` allo stato (50 → 53 dim). Richiede heading da Gazebo.

**Opzione B (più potente):** stack degli ultimi 3 frame LIDAR (50×3 = 150 dim). Come DQN su Atari.

**Opzione C (più potente ma complessa):** sostituire MLP con DRQN (LSTM). Richiede refactor del training loop.

**Raccomandazione:** iniziare con Opzione A (semplice, leggibile).

---

### 6. Prioritized Experience Replay (PER)

**Perché:** Con crash rate 99.6%, il buffer da 100k contiene ~3% transizioni di crash. Uniform sampling le sottocampiona. PER campiona proporzionalmente al TD error (Schaul et al. 2015).

**Impatto atteso:** moderato se abbinato a reward shaping (punto 2). Più critico se si mantiene reward binario.

**Come:** libreria `prio_replay_buffer` o implementazione custom con sum-tree. ~100 righe.

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

## Roadmap sintetica

```
ITERAZIONE A (fix immediati, ~2h coding):
  fix 4 (Huber + clip) + fix 7 (MAX_STEPS) → più stabile
  fix 8 (re-test) → dati puliti

ITERAZIONE B (goal + shaping, ~1 giorno):
  fix 1 (goal in stato) + fix 2 (reward shaping) → task corretto
  → stimato crash rate Maze 2: 30-50% in 3000 ep

ITERAZIONE C (generalizzazione, ~1 giorno):
  fix 3 (multi-maze) + fix 5 (contesto temporale)
  → stimato crash rate Maze 2: <30%, Maze 1: ~40%
```

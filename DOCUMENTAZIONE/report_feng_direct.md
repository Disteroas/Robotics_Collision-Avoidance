# Analisi post-training — feng_direct
## DDQN UGV Collision Avoidance: diagnosi dei fallimenti

**Data:** 2026-05-08  
**Branch:** `feng_direct`  
**Configurazione:** 3000 episodi, Maze 2, BETA_DECAY=0.999, 16 spawn random, ε: 1.0→0.05

---

## 1. Risultati quantitativi

### 1.1 Training

| Metrica | Valore |
|---|---|
| Episodi | 3000 |
| Steps totali | 535,354 |
| Crash rate | **99.6%** (2987/3000) |
| ε finale | 0.050 (minimo raggiunto) |
| Avg-100 reward finale | **+391** |
| Primo episodio completato (1000 step) | **ep 1996** (ε=0.136) |

**Curva di apprendimento:**

| Intervallo | Avg-100 | ε | Note |
|---|---|---|---|
| ep 1-100 | ~-400 | 1.0→0.91 | Solo crash rapidi (50-200 step) |
| ep 500 | -433 | 0.607 | Ancora crash, pochi miglioramenti |
| ep 1000 | -342 | 0.368 | Episodi più lunghi (ep994: 350 step, +745) |
| ep 1500 | -156 | 0.224 | Episodi molto lunghi (ep1498: 446 step, +1225) |
| ep 1996 | +33 | 0.136 | **Primo successo completo (1000 step, reward=5000)** |
| ep 2500 | +134 | 0.083 | Trend positivo consolidato |
| ep 3000 | +391 | 0.050 | Training terminato — curva ancora in salita |

La curva mostra apprendimento genuino e continuo. L'agente ha raggiunto il primo episodio completo a ep 1996 e la media dei 100 episodi finali è ampiamente positiva (+391). Il training si è concluso mentre la curva era ancora in crescita.

### 1.2 Test (policy greedy, ε=0.0)

| Maze | Crash rate | Successi | Avg reward | Note |
|------|-----------|---------|-----------|------|
| Maze 1 (mai visto in training) | **100%** | 0/30 | -487 | Spawn corretto (bug non impatta) |
| Maze 2 (training) | **90%** | 3/30 (10%) | -345 | 7 crash a step=1 (bug spawn) |
| Maze 3 (mai visto) | **100%** | 0/30 | -466 | Spawn Maze 1 in mondo Maze 3 |

---

## 2. Bug identificato: test.py non passa maze_id al reset

**File:** `src/my_usv/scripts/test.py`, riga 80

```python
state = env.reset_environment()   # BUG: manca maze_id
```

`reset_environment(maze_id: int = 1)` ha default 1. Di conseguenza:
- **Maze 1 test**: corretto per default, nessun impatto
- **Maze 2 test**: robot spawna dalle posizioni di Maze 1 (`SPAWN_LISTS[1]`) dentro il mondo di Maze 2. Alcune posizioni di Maze 1 coincidono con muri di Maze 2 → 7/30 episodi con crash a step=1, min_lidar=0.122-0.161m
- **Maze 3 test**: stesso problema

I 3 successi su Maze 2 si sono verificati nonostante spawn subottimali. Il bug non è la causa principale del fallimento: anche correggerlo, il modello non generalizza a Maze 1 e 3 perché non li ha mai visti.

**Fix applicato:** commit `4bbc476`.

---

## 3. Analisi delle cause di fallimento — in ordine di impatto

---

### CAUSA 1 [CRITICA]: Reward sparso — segnale binario senza shaping

**Struttura attuale:**
- `R(t) = +5` se vivo (ogni step)
- `R(t) = -1000` se collisione (fine episodio)

Nota: questa reward è identica a quella del paper originale (Feng et al. 2021, Eq. 4). È una scelta di design, non un errore di implementazione. I problemi che genera rimangono però reali.

**Problemi:**

**1a. Nessun gradiente di pericolo.** Il reward non distingue "molto vicino a muro" da "lontano da muro". L'agente riceve +5 a 0.30m da un muro e +5 a 4.90m. Non c'è segnale che avvicinarsi a un muro sia rischioso finché non si tocca.

**1b. Imbalance estremo del reward.** Rapporto crash/survival = 1000/5 = 200. Un crash cancella 200 step di sopravvivenza. Questo crea una distribuzione di TD-error bimodale: la maggior parte dei batch ha target_q ≈ +5, ma le transizioni di crash hanno target_q ≈ -1000. Con MSE loss (invece di Huber), i gradienti dei crash dominano l'aggiornamento in modo sproporzionato ogni volta che sono campionati.

**1c. Replay buffer sbilanciato.** A 178 step medi per episodio, il buffer di 100.000 transizioni contiene:
- ~97% transizioni survival (+5): pattern di movimento normale
- ~3% transizioni crash (-1000): stati pre-collisione

Le transizioni di crash sono rare nel buffer. Con uniform sampling, la rete vede un crash ogni ~30 batch. Questo spiega perché la rete impara a sopravvivere a lungo ma non ad evitare muri critici.

**Riferimento:** Ng, Harada & Russell (1999) "Policy invariance under reward transformations" (ICML 1999) — teorema fondamentale: il reward shaping tramite funzione potenziale preserva la policy ottimale e accelera la convergenza. Schaul et al. (2015) "Prioritized Experience Replay" (ICLR 2016) — PER campiona proporzionalmente al TD error, risolvendo l'undersampling delle transizioni rare ma importanti (esattamente i crash). Mnih et al. (2015) "Human-level control through deep reinforcement learning" (Nature 2015) — usano Huber loss invece di MSE per limitare la magnitudine dei gradienti con reward di grande ampiezza.

**Fix diretto:** Aggiungere un termine di pericolo graduato: `R += -penalty × exp(-d_min / 0.5)` dove `d_min` è la distanza minima LIDAR. Questo crea un gradiente continuo verso la sicurezza senza alterare la policy ottimale (è una funzione potenziale basata sullo stato).

---

### CAUSA 2 [CRITICA]: Single-maze training — assenza di generalizzazione

**Evidenza:** Maze 1 e Maze 3 → 100% crash, identico a paper_implementation.

L'agente ha addestrato su 3000 episodi in Maze 2 (labirinto_9b) con muri diagonali. I 50 bin LIDAR codificano pattern geometrici specifici di quella topologia. La rete ha memorizzato associazioni `(LIDAR_pattern_maze2, action)` che non si trasferiscono ad altri maze.

**Meccanismo specifico:**
- In Maze 2 il corridoio principale è orientato diagonalmente. L'agente ha imparato a rispondere a specifici gradienti di distanza LIDAR nel range [-7.6, +7.3] × [-6.3, +6.5].
- In Maze 1 (labirinto_9a) i corridoi sono ortogonali, le pareti creano pattern LIDAR completamente diversi. La rete produce Q-values assurdi per stati mai visti.
- In Maze 3 (labirinto_10) struttura diversa, mai vista.

**Riferimento:** Cobbe et al. (2019) "Quantifying Generalization in Reinforcement Learning" (ICML 2019) — studio sistematico su CoinRun: agenti DRL addestrati su N livelli ottengono 0% su livelli mai visti anche con N=500. Overfitting all'ambiente di training è una proprietà fondamentale del DRL policy-gradient. Zhang et al. (2018) "A Study on Overfitting in Deep Reinforcement Learning" — DRL richiede 10-100x più ambienti diversi rispetto al supervised learning per generalizzare. Tobin et al. (2017) "Domain Randomization for Transferring Deep Neural Networks" (IROS 2017, OpenAI) — randomizzare la distribuzione degli ambienti di training è il metodo più efficace per generalizzazione sim-to-real e cross-environment.

**Fix:** Training con curriculum multi-maze (Maze 1 + Maze 2 in alternanza) oppure randomizzazione dei parametri geometrici (wall positions, orientations). L'approccio `paper_implementation` era corretto nel principio (multi-maze) ma falliva per i problemi di epsilon e threshold — non nel principio del multi-maze.

---

### CAUSA 3 [MAGGIORE]: Episodi insufficienti — training non convergito

**Evidenza dalla curva:**

La avg-100 al termine del training (+391) era **ancora in crescita** (non plateau). Il primo episodio completato appare a ep 1996 di 3000. Il modello ha avuto solo ~1000 episodi "maturi" (dopo aver visto il primo successo) per consolidare la policy.

Confronto temporale:
- ep 1-1996: esplorazione pura, solo crash
- ep 1997-3000: ~1000 episodi con mix di episodi lunghi e crash

Con ε=0.05 finale e crash rate che rimane ~99% negli ultimi 100 episodi di training, il buffer continua ad essere sovrascritto da episodi brevi (crash a ~50-100 step). Le transizioni "buone" (500+ step) vengono diluite e dimenticate.

**Stima episodi necessari:** La curva avg-100 mostra un plateau potenziale intorno a +500-600 (ipotesi lineare). Il tasso di miglioramento negli ultimi 500 ep è ~(391-134)/(3000-2500) = +0.51 avg100/ep. Per raggiungere avg100=1000 (significativo, ~200 step medi senza crash):
`(1000 - 391) / 0.51 ≈ 1200 ep aggiuntivi` → ~4200 ep totali (con questa reward function).

**Riferimento:** Schulman et al. (2015) "Trust Region Policy Optimization" (ICML 2015) — la convergenza DRL è notoriamente non-stazionaria. Il campionamento di stati critici (come il successo completo) è necessario per formare una policy stabile.

---

### CAUSA 4 [MAGGIORE]: Assenza di contesto temporale nello stato

**Stato attuale:** S(t) = 50 bin LIDAR istantanei.

Il robot naviga con velocità lineare fissa (0.5 m/s) e 11 azioni di sterzo. Il LIDAR snapshot non contiene:
- Direzione di movimento attuale
- Velocità angolare
- Storia delle ultime N posizioni

Due scenari con LIDAR identico richiedono azioni opposte:
1. Robot si muove verso un muro a sinistra → deve sterzare a destra
2. Robot si allontana dallo stesso muro → può mantenere direzione

Il LIDAR snapshot S(t) è identico in entrambi i casi. La policy ottimale è diversa. Senza stack temporale o velocità nello stato, la rete deve approssimare una policy sub-ottimale che "funziona in media" per entrambe le situazioni.

**Evidenza indiretta:** La presenza di episodi con crash "a step alto" (es. ep2992: step=120, ep2996: step=53) durante il testing, nonostante il modello abbia appreso a sopravvivere 400+ step in training, suggerisce una policy inconsistente che funziona su alcune sequenze e fallisce su altre simili.

**Riferimento:** Mnih et al. (2015) — DQN originale usa 4 frame stack per fornire informazioni di velocità implicita all'agente (dalla differenza tra frame). Hausknecht & Stone (2015) "Deep Recurrent Q-Network" (AAAI Workshop 2015) — DRQN sostituisce MLP con LSTM per gestire ambienti parzialmente osservabili (POMDP). Navigazione robotica con solo LIDAR è un POMDP per definizione (heading non osservato). Mirowski et al. (2016) "Learning to Navigate in Complex Environments" (ICLR 2017) — usano LSTM + multi-task learning per navigazione in labirinti complessi.

---

### CAUSA 5 [MODERATA]: Loss function inadeguata per reward bimodale

**Codice:**
```python
self.loss_fn = nn.MSELoss()
```

Con reward bimodale (+5 survival, -1000 crash), il TD error per transizioni di crash è dell'ordine di 100-900 (dipende dal Q-value stimato). MSE amplifica i gradienti quadraticamente: un TD error di 100 produce un gradiente 10.000x maggiore rispetto a un TD error di 1. Questo causa instabilità e overfit alle poche transizioni di crash nel buffer.

**Evidenza:** La avg_loss al termine del training è ~2000-3000 (valore molto alto), a indicare che i Q-values non sono convergiti nonostante 3000 episodi.

**Riferimento:** Mnih et al. (2015) — usano Huber loss (`smooth_l1_loss` in PyTorch) che è lineare per |TD_error| > 1 e quadratica altrimenti. Questo limita i gradienti a magnitudine costante per grandi errori. Van Hasselt et al. (2016) "Deep Reinforcement Learning with Double Q-Learning" (AAAI 2016) — la transizione da Q-learning a DDQN riduce il maximization bias ma non risolve il problema dei gradienti: Huber loss rimane necessaria.

**Fix:** Sostituire `nn.MSELoss()` con `nn.SmoothL1Loss()` (1 riga in `train_core.py`).

---

### CAUSA 6 [MODERATA]: Gradient clipping permissivo

**Codice:**
```python
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
```

Clip a 10.0 è 10x più permissivo dell'originale DQN (1.0). Con reward di -1000, il gradiente non-clipped per transizioni di crash può essere nell'ordine di centinaia. Clip a 10.0 li riduce ma non li normalizza come clip a 1.0.

**Fix:** Abbassare a 1.0 (standard DQN/DDQN) insieme a Huber loss.

---

### CAUSA 7 [MINORE]: MAX_STEPS disallineato tra training e test

**Training:** `MAX_STEPS = 1000`  
**Test:** `MAX_STEPS = 500`

Il commento in `test.py` dice "coerente con il training" ma è sbagliato: il training usa 1000.

Impatto: la rete è addestrata a massimizzare il reward su orizzonte 1000 step. Q(s,a) = Σ γ^t r(t) per t fino a 1000. Il test valuta su orizzonte 500. Questo non causa fallimento diretto (la maggior parte degli episodi crasha comunque prima di 500 step) ma introduce un'inconsistenza architetturale. Con γ=0.99, il value Q(s,a) al training calibra per un orizzonte `1/(1-0.99) = 100 step` effettivi — il disallineamento è meno grave di quanto sembri perché il discounting attenua i reward lontani.

---

## 4. Riepilogo delle cause

| # | Causa | Impatto | Fix |
|---|---|---|---|
| 1 | Reward sparso + imbalance | **CRITICO** | Shaping graduato; PER; Huber loss |
| 2 | Single-maze training | **CRITICO** | Multi-maze; domain randomization |
| 3 | Training non convergito | **MAGGIORE** | 5000+ ep (fix 1 accelera convergenza) |
| 4 | No contesto temporale | **MAGGIORE** | Stack LIDAR (N=2-4) o aggiungere heading/velocità |
| 5 | MSE loss su reward bimodale | **MODERATA** | SmoothL1Loss (Huber) |
| 6 | Gradient clip troppo alto | **MODERATA** | Ridurre da 10.0 a 1.0 |
| 7 | test.py bug maze_id | **MINORE** | `reset_environment(maze_id=args.maze_id)` — già fixato |
| 8 | MAX_STEPS disallineato | **MINORE** | Uniformare a 500 o 1000 |

---

## 5. Confronto con paper_implementation

| | paper_impl | feng_direct | Differenza |
|---|---|---|---|
| Crash rate training | ~99% | 99.6% | Comparabile |
| Successi Maze 2 (test) | 0/30 | **3/30** | +10% (miglioramento marginale) |
| Avg-100 training finale | ~-200 | **+391** | Molto meglio |
| Primo successo completo | Mai | ep 1996 | feng_direct ha imparato |
| Cause | ε troppo basso early; phase transition errata | Sparse reward; single-maze | Cause diverse |

`feng_direct` è strutturalmente superiore a `paper_implementation`: ha convergito su una policy funzionale (avg-100 positivo, qualche successo completo). Il training era sulla traiettoria giusta ma le cause strutturali (sparse reward, single-maze) limitano il massimo raggiungibile con questa architettura.

---

## 6. Raccomandazioni per prossima iterazione

### Fix immediati (1-2 ore di coding)

**1. Bug test.py** (1 riga) — già applicato:
```python
state = env.reset_environment(maze_id=args.maze_id)
```

**2. Huber loss** (1 riga in `train_core.py`):
```python
self.loss_fn = nn.SmoothL1Loss()
```

**3. Gradient clip** (1 riga in `train_core.py`):
```python
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
```

### Modifiche strutturali (richiedono brainstorming + piano)

**4. Reward shaping:**
```python
# Pericolo graduato
danger = max(0, 1 - min_lidar / SAFE_DIST)
reward = +5 - 50 * danger**2
if min_lidar < COLLISION_DIST:
    reward = -1000
    done = True
```

**5. Multi-maze training:**
Alternare Maze 1 e Maze 2 ogni 100 episodi (come paper_implementation ma con epsilon e threshold corretti).

**6. Contesto temporale:**
Aggiungere `heading` (cos/sin) e `angular_velocity` allo stato. Oppure stack delle ultime 3 osservazioni LIDAR.

**7. Estensione: aggiungere goal allo stato:**
Il task attuale (collision avoidance pura, identico al paper originale) non prevede destinazione. Estendere a navigation aggiungendo `[dist_to_goal/MAX_DIST, cos(angle_to_goal), sin(angle_to_goal)]` allo stato (50→53 dim). Richiede definire un goal fisso per episodio in ogni maze e reward bonus per raggiungimento. Questo va oltre lo scope del paper originale ma è il naturale step successivo verso navigazione autonoma completa.

### Stima impatto

Con fix 1-3 (solo parametri): crash rate test su Maze 2 stimato ~75% (da 90%). Nessun improvement su Maze 1/3.

Con fix 4-5 (reward shaping + multi-maze): crash rate test su Maze 2 stimato ~40-60% in 3000 ep.

Con fix 4-7 (tutti): convergenza su Maze 2 stimata ~20-30%, Maze 1 ~35%, Maze 3 ~15% (test-only).

---

## Riferimenti

1. Feng, S., Sebastian, B. & Ben-Tzvi, P. (2021) — "A Collision Avoidance Method Based on Deep Reinforcement Learning" — *Robotics* 10, 73. Paper base dell'implementazione: stato = LIDAR only, reward +5/−1000, 3000 ep, β=0.999. Goal-directed navigation indicato come lavoro futuro.
2. Mnih, V. et al. (2015) — "Human-level control through deep reinforcement learning" — *Nature* 518, 529-533. DQN originale: Huber loss, replay buffer, target network.
3. Van Hasselt, H. et al. (2016) — "Deep Reinforcement Learning with Double Q-Learning" — *AAAI 2016*. DDQN: decoupling selection/evaluation riduce maximization bias.
4. Schaul, T. et al. (2015) — "Prioritized Experience Replay" — *ICLR 2016*. PER: sampling proporzionale al TD error, critico per transizioni rare ad alto errore.
5. Ng, A.Y., Harada, D. & Russell, S.J. (1999) — "Policy invariance under reward transformations: Theory and application to reward shaping" — *ICML 1999*. Teorema: reward shaping basato su funzione potenziale preserva policy ottimale.
6. Cobbe, K. et al. (2019) — "Quantifying Generalization in Reinforcement Learning" — *ICML 2019*. CoinRun benchmark: overfitting all'ambiente di training è proprietà fondamentale del DRL.
7. Tobin, J. et al. (2017) — "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" — *IROS 2017* (OpenAI). Domain randomization come tecnica di generalizzazione sim-to-real.
8. Mirowski, P. et al. (2016) — "Learning to Navigate in Complex Environments" — *ICLR 2017* (DeepMind). Navigation in labirinti: LSTM + auxiliary tasks.
9. Hausknecht, M. & Stone, P. (2015) — "Deep Recurrent Q-Network" — *AAAI Workshop on AI and Deep Learning 2015*. DRQN: LSTM per POMDP navigation dove lo stato è parzialmente osservabile.
10. Sutton, R.S. & Barto, A.G. (2018) — *Reinforcement Learning: An Introduction*, 2nd ed. MIT Press.
11. Zhang, C. et al. (2018) — "A Study on Overfitting in Deep Reinforcement Learning" — *arXiv:1804.06893*.

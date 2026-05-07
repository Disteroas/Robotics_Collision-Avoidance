# Report Paper — Spawn Points e Generalizzazione nel DRL per Mobile Robot

**Progetto di riferimento:** USV DDQN Collision Avoidance (LiDAR, Gazebo, ROS2)  
**Domanda guida:** Come scegliere gli spawn point? Il training su mappa singola funziona?

---

## Panoramica dei 3 Paper

| Paper | Autori | Anno | Citazioni | Rilevanza |
|---|---|---|---|---|
| *Reverse Curriculum Generation for RL* | Florensa et al. | CoRL 2017 | ~500 | ⭐⭐⭐⭐⭐ |
| *A Study on Overfitting in Deep RL* | Zhang et al. | arXiv 2018 | ~400 | ⭐⭐⭐⭐⭐ |
| *Enhancing DRL-based Robot Nav. Generalization through Scenario Augmentation* | Wang et al. | arXiv 2025 | nuovo | ⭐⭐⭐⭐⭐ |

---

## Paper 1 — Reverse Curriculum Generation for Reinforcement Learning

**Florensa, Held, Wulfmeier, Zhang, Abbeel — CoRL 2017 — arXiv:1707.05300**

### Il Problema

Nei task goal-oriented con reward sparso (es. raggiungere una zona senza collisioni), il robot parte da stati così lontani dall'obiettivo che non riceve mai un segnale di reward positivo nelle prime fasi del training. Senza questo segnale, la rete non impara nulla. Il problema si amplifica con spawn casuali uniformi: molti spawn finiscono in zone irrecuperabili, sprecando episodi senza nessun apprendimento.

Il contesto è esattamente il nostro: robot in navigazione 2D su maze, reward sparso (sopravvivi o crashy), stato continuo.

### La Soluzione: Reverse Curriculum

L'idea chiave è capovolgere la logica del training. Invece di partire da qualunque posizione casuale, si parte **vicino al goal** (zona sicura) e si espande progressivamente verso l'esterno man mano che il robot impara.

**L'algoritmo in 4 passi:**

1. **Inizializzazione**: spawn solo vicino alla zona goal (il robot ha alta probabilità di avere successo → segnale di reward immediato)
2. **Espansione**: da ogni stato "buono" (episodio completato con successo), si eseguono brevi *random walk* in spazio delle azioni (*Brownian motion*, orizzzonte T_B passi). Gli stati raggiunti diventano nuovi spawn candidati
3. **Selezione dei "good starts"**: si trattengono solo gli spawn con `R_min < R(π, s₀) < R_max` — né troppo facili (già padroneggiati) né troppo difficili (reward = 0). Solo quelli "al limite" della competenza attuale
4. **Replay buffer di spawn**: si mantiene un buffer degli spawn buoni precedenti per evitare il catastrophic forgetting sulle zone già padroneggiate

```
Pseudocode semplificato:
  starts = [goal_state]
  for ogni iterazione:
      new_starts = BrownianWalk(starts, T_B steps)  ← esplora vicino ai buoni
      all_starts = new_starts + sample(starts_old)   ← più replay buffer
      train(policy, all_starts)
      starts = select(all_starts, R_min < R < R_max) ← solo i "buoni"
      starts_old.append(starts)
```

### Risultati Quantitativi

| Task | Metodo | Successo |
|---|---|---|
| Point-mass maze (2D nav.) | Uniform spawn | ~30%, alta varianza |
| Point-mass maze (2D nav.) | Reverse Curriculum | ~95%, stabile |
| Ant maze (robot 4 zampe) | Uniform spawn | ~50%, lento |
| Ant maze (robot 4 zampe) | Reverse Curriculum | ~90%, 3-5× più veloce |
| Ring on Peg (7-DOF arm) | Uniform spawn | ~10% |
| Ring on Peg (7-DOF arm) | Reverse Curriculum | ~80% |
| Key insertion (7-DOF arm) | Uniform spawn | ~2% |
| Key insertion (7-DOF arm) | Reverse Curriculum | ~60% |

Il risultato più importante: **uniform spawn fallisce completamente sui task difficili**. Il reverse curriculum li risolve tutti.

### Applicazione al Progetto USV

Il problema attuale su Maze 2 è esattamente questo: lo spawn fisso `x=-6, y=0, yaw=0` punta sempre nella stessa direzione verso un muro, e il robot crasha sempre dopo ~61 step. Con uno spawn random uniforme, molte posizioni di partenza sarebbero comunque problematiche (vicino alle pareti, orientamento sbagliato).

Con il reverse curriculum:
- Si partirebbe da un punto sicuro al centro del maze (lontano da tutti i muri)
- Si espanderebbe gradualmente verso le zone difficili (angoli, corridoi stretti)
- Il robot imparerebbe prima a stare lontano dai muri in zona aperta, poi ad affrontare le geometrie complesse

**Limite per il nostro stack**: richiede che Gazebo possa fare spawn in posizioni arbitrarie (`ros2 service call /set_entity_state`). Se implementabile, è la soluzione migliore.

---

## Paper 2 — A Study on Overfitting in Deep Reinforcement Learning

**Zhang, Vinyals, Munos, Bengio — arXiv:1804.06893 (Google Brain) — 2018**

### Il Problema

Questo paper risponde direttamente alla domanda: *"Se uso spawn random sulla stessa mappa, evito l'overfitting?"*

Risposta: **no**. E lo dimostra sperimentalmente in modo rigoroso.

Il paper usa un ambiente maze configurabile (simile al nostro) e testa sistematicamente se le tecniche standard per aggiungere stocasticità durante il training prevengono l'overfitting:

- Spawn casuali (RAND-SPAWN)
- Azioni sticky (STICKY-ACTION)
- Policy stocastica
- Combinazioni delle precedenti

### Risultati Principali

**Finding #1: Gli agenti DRL memorizzano le mappe**

Con 10 mappe di training, gli agenti raggiungono reward ottimale in training ma reward vicino a 0 su mappe di test. **Lo stesso agente che performa perfettamente in training fallisce completamente su mappe nuove.**

Con 10.000 mappe di training, il gap diminuisce ma non scompare.

| Training levels | Train reward | Test reward (BASIC) | Test reward (TUNNEL) |
|---|---|---|---|
| 10 | 2.1 (ottimale) | ~0.5 | ~-0.5 |
| 100 | 2.1 | ~1.2 | ~0.3 |
| 1.000 | 2.1 | ~1.8 | ~0.8 |
| 10.000 | 2.1 | ~2.0 | ~1.5 |

**Finding #2: Spawn random non prevengono l'overfitting**

Il paper testa RAND-SPAWN come regolarizzatore durante il training. Risultato: gli agenti migliorano leggermente il test performance, **ma sono ancora in grado di memorizzare anche mappe con reward completamente casuali**. Il training reward rimane ottimale anche con p=0.5 di reward flipping (caso impossibile da generalizzare per costruzione).

> *"Random starts help to improve the test performances a bit. However, the agents are still able to fit even random training levels almost optimally."*

**Finding #3: La tecnica non funziona neanche come detection**

Se si aggiunge RAND-SPAWN durante la *valutazione* (non il training), l'overfitting non viene rilevato: gli agenti overfittati mantengono alto il score anche con spawn casuali sul training set, ma falliscono ugualmente sul test set.

### Raccomandazione degli Autori

> *"Isolation of training and test data is recommended even for noisy and non-deterministic environments."*

Serve separazione train/test a livello di **mappa**, non solo di spawn point. Questo è esattamente il motivo per cui il nostro Maze 3 come held-out test set è la scelta corretta.

### Applicazione al Progetto USV

Questo paper conferma due cose critiche per il nostro progetto:

1. **Il training su Maze 2 con spawn fisso è il caso peggiore possibile**: spawn fisso + mappa singola = memorizzazione garantita, nessuna generalizzazione
2. **Anche passando a spawn random su Maze 2, il problema non è risolto**: il robot memorizzerà la geometria di Maze 2 indipendentemente da dove parte

La soluzione indicata è usare più mappe diverse (che nel nostro caso abbiamo già: Maze 1 e Maze 2 in alternanza) E misurare sempre su Maze 3 come held-out set (che facciamo già).

---

## Paper 3 — Enhancing DRL-based Robot Navigation Generalization through Scenario Augmentation

**Wang, Tan, Yang, Wang, Shen, Huang, Zhang — arXiv:2503.01146 — 2025**

### Il Problema

**Questo è il paper più vicino al nostro problema specifico** (mobile robot, LiDAR, collision avoidance, Gazebo, ROS2).

Gli autori partono dalla stessa osservazione dei paper precedenti: i robot DRL allenati su uno o due ambienti fissi falliscono in ambienti nuovi. Ma invece di richiedere più mappe (costose da costruire), propongono un metodo per fare *data augmentation sull'osservazione* che simula virtualmente molti ambienti diversi.

> *"Limited training scenarios represent the primary factor behind these undesired behaviors."*

### La Soluzione: Scenario Augmentation

L'idea è sorprendentemente semplice: **trasformare l'osservazione LIDAR prima di darla alla rete**, simulare un'azione in questo spazio trasformato, e rimappare l'azione nello spazio reale.

**Meccanismo in 3 passi:**

```
1. Observation mapping:
   o_real → [trasformazione T] → o_imagined
   (es. rotazione di 90°, riflessione, inversione sinistra-destra del LIDAR)

2. Action generation:
   o_imagined → [policy π] → a_imagined

3. Action remapping:
   a_imagined → [trasformazione inversa T⁻¹] → a_real
```

In pratica: se la policy vede i 50 raggi LIDAR ruotati di 90°, impara a fare collision avoidance da quella prospettiva. Questo equivale ad allenare il robot come se stesse navigando da un'angolazione diversa, **senza cambiare il mondo fisico in Gazebo**.

**Trasformazioni utilizzabili:**
- Rotazione dei raggi LIDAR (simula orientamento diverso dello spawn)
- Riflessione sinistra-destra (simula geometrie speculari)
- Inversione degli indici dei raggi (come guardare indietro)
- Combinazioni delle precedenti

### Risultati

Il framework è testato su **mobile robot con LIDAR in Gazebo** — identico al nostro setup:

| Metodo | Train env performance | Test env (unseen) performance | Riduzione collisioni |
|---|---|---|---|
| Baseline (single scenario) | Alta | Bassa | — |
| + Scenario Augmentation | Alta | **Sostanzialmente più alta** | Significativa |
| Real-world deployment | Near-optimal trajectories | Riduzione significativa del tempo di navigazione | ✅ |

Gli autori riportano "near-optimal trajectories with significantly reduced navigation time in real-world applications" usando il modello allenato solo in simulazione.

### Applicazione al Progetto USV

Questo è il fix più semplice da implementare nel nostro stack attuale: **non richiede più mappe, non richiede modifiche a Gazebo, non richiede cambio di architettura**.

Basta modificare `step_action()` in `usv_env.py` per applicare trasformazioni casuali allo scan LIDAR prima di restituire lo stato:

```python
import numpy as np

def _augment_scan(self, scan: np.ndarray) -> np.ndarray:
    """
    Scenario augmentation: trasforma l'osservazione LIDAR
    simulando virtual mappe diverse senza cambiare la fisica.
    """
    mode = np.random.randint(0, 4)
    if mode == 0:
        return scan                          # identità (30% del tempo)
    elif mode == 1:
        return np.roll(scan, 12)             # rotazione +90° (25 beams ≈ 90°)
    elif mode == 2:
        return np.roll(scan, -12)            # rotazione -90°
    elif mode == 3:
        return np.flip(scan).copy()          # riflessione sinistra-destra
    return scan

def get_state(self) -> np.ndarray:
    raw = (self.current_scan / LIDAR_MAX_RANGE).copy()
    if self.training_mode:               # disabilitato durante il test
        raw = self._augment_scan(raw)
    return raw
```

**Attenzione:** la trasformazione va applicata coerentemente. Se la scan viene ruotata di 90° nell'osservazione, anche l'azione prodotta dalla rete si riferisce a quello spazio ruotato. Il remapping dell'azione (sterzo) deve essere invertito di conseguenza:

```python
# Se scan ruotata di +90° → l'azione di "dritto" diventa "sinistra" nel mondo reale
# Questo va gestito nel remapping: per semplicità, si può usare solo flip sinistra-destra
# dove l'azione si mappa naturalmente come: a_real = ACTION_DIM - 1 - a_imagined
```

---

## Sintesi Comparativa

| | Florensa 2017 | Zhang 2018 | Wang 2025 |
|---|---|---|---|
| **Problema** | Spawn casuali uniformi sprecano episodi in zone irrecuperabili | Spawn casuali non prevengono memorizzazione della mappa | Singolo scenario → policy fragile su ambienti nuovi |
| **Soluzione** | Curriculum di spawn espanso dal goal verso l'esterno | Separazione train/test a livello di mappa | Data augmentation sull'osservazione LIDAR |
| **Robot** | Quadrupede, 7-DOF arm, point-mass | Gridworld maze agent | **Mobile robot con LiDAR** ← identico al nostro |
| **Simulator** | MuJoCo | Custom gridworld | **Gazebo** ← identico al nostro |
| **Difficoltà implementazione** | Alta (richiede spawning arbitrario in Gazebo) | N/A (è un paper diagnostico) | **Bassa (solo codice Python)** |
| **Applicabilità immediata** | Media | Diagnostica (conferma i problemi) | **Alta — modificabile in 1 ora** |

---

## Idee per il Progetto — Ordine di Priorità

### 🟢 Idea 1 — Scenario Augmentation (da Wang 2025) — Implementazione immediata

Applica trasformazioni casuali allo scan LIDAR in `usv_env.py` durante il training. Nessuna modifica all'infrastruttura Docker/Gazebo.

**Effort**: 30 minuti di codice.  
**Beneficio atteso**: il robot "vede" geometrie diverse ad ogni episodio senza cambiare il mondo fisico → generalizzazione su Maze 3 migliorata.  
**Rischio**: se il remapping azione non è implementato correttamente, il training diverge. Iniziare solo con la riflessione sinistra-destra (flip) dove il remapping è semplice.

### 🟡 Idea 2 — Spawn Random reali su Maze 2 (da Zhang 2018)

Implementare spawn in posizioni diverse all'interno di Maze 2 usando il servizio ROS2 `set_entity_state` invece di `reset_world`. Richiederebbe di conoscere le coordinate valide (non dentro le mura) e randomizzare `(x, y, yaw)` ad ogni reset.

**Effort**: medio (mappa del maze → lista coordinate valide → selezione casuale).  
**Beneficio atteso**: elimina il comportamento deterministico (crash sempre a step 61). Il robot vede più della geometria del maze.  
**Limite (da Zhang 2018)**: anche con spawn random su mappa singola, il robot memorizzerà comunque la mappa. Utile ma non sufficiente.

### 🔴 Idea 3 — Reverse Curriculum (da Florensa 2017) — Implementazione avanzata

Costruire un curriculum di spawn che parte dal centro del maze (zona libera) ed espande verso le zone difficili. Richiede di definire una "goal zone" (navigazione senza crash per N step), un meccanismo di random walk da essa, e selezione degli spawn "good" per return range.

**Effort**: alto (rifacimento del loop di training + integrazione con Gazebo spawn API).  
**Beneficio atteso**: massimo — il robot impara le zone difficili del maze in modo ordinato invece che casuale.  
**Quando ha senso**: se dopo aver implementato Idea 1 e Idea 2 il robot continua a crashare su Maze 2 sistematicamente.

---

## Riferimenti

- Florensa C., Held D., Wulfmeier M., Zhang M., Abbeel P. *Reverse Curriculum Generation for Reinforcement Learning.* CoRL 2017. arXiv:1707.05300
- Zhang C., Vinyals O., Munos R., Bengio S. *A Study on Overfitting in Deep Reinforcement Learning.* arXiv:1804.06893 (2018)
- Wang S., Tan M., Yang Z., Wang X., Shen X., Huang H., Zhang W. *Enhancing Deep Reinforcement Learning-based Robot Navigation Generalization through Scenario Augmentation.* arXiv:2503.01146 (2025)

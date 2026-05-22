# Analisi strategica — DDQN Collision Avoidance USV

**Data:** 2026-05-21
**Branch di riferimento:** `ddqn_en_20_05`
**Autore della sintesi:** analisi condivisa (da discutere col gruppo)
**Scopo del documento:** mettere nero su bianco, in modo completo e ordinato, il ragionamento che ci ha portato dalla scoperta della varianza fino al piano di lavoro proposto. È pensato per essere letto dai colleghi del gruppo prima di decidere come procedere. Non è una sintesi: è il quadro completo, con i numeri, la matematica, la letteratura e le scelte.

---

## Indice

1. [Punto di partenza: cosa abbiamo addestrato](#1-punto-di-partenza-cosa-abbiamo-addestrato)
2. [La scoperta: varianza enorme tra run identiche](#2-la-scoperta-varianza-enorme-tra-run-identiche)
3. [La causa: nessun seed fissato](#3-la-causa-nessun-seed-fissato)
4. [Le conseguenze metodologiche](#4-le-conseguenze-metodologiche)
5. [Diagnosi dei fallimenti: due classi distinte](#5-diagnosi-dei-fallimenti-due-classi-distinte)
6. [La letteratura e come si applica a noi](#6-la-letteratura-e-come-si-applica-a-noi)
7. [Le domande di inquadramento e le istruzioni dei prof](#7-le-domande-di-inquadramento-e-le-istruzioni-dei-prof)
8. [La decisione strategica: il paper a 3 parti](#8-la-decisione-strategica-il-paper-a-3-parti)
9. [Il piano operativo: spina dorsale + due tracce in parallelo](#9-il-piano-operativo-spina-dorsale--due-tracce-in-parallelo)
10. [Cosa NON fare](#10-cosa-non-fare)
11. [Riferimenti bibliografici](#11-riferimenti-bibliografici)

---

## 1. Punto di partenza: cosa abbiamo addestrato

Il progetto riproduce e aggiorna **Feng et al. 2021** (DDQN su LiDAR in Gazebo/ROS2) applicato a un USV simulato che deve attraversare labirinti evitando collisioni.

Configurazione attuale (Round 2, reward "R-α"):

| Componente | Valore |
|---|---|
| Algoritmo | DDQN (Double DQN) |
| Stato | 152-dim = 50 bin LIDAR × 3 frame stack + 2 heading (cos/sin yaw) |
| Azioni | 11 velocità angolari discrete da −0.8 a +0.8 rad/s |
| Velocità lineare | **fissa** a 0.5 m/s |
| Reward | +5 base, penalità sterzata `0.02·|a−5|`, `space_bonus = 2·mean(scan)/5`, front danger `10·severity²` (<1.5m), side danger `3·severity²` (<0.45m), collisione **−1000** (<0.25m, termina episodio) |
| γ | 0.99 |
| Episodi | 5000 (multi-maze M1+M2, pattern 1:2) |

L'addestramento è stato lanciato **due volte, su due PC diversi, con codice e configurazione identici** (run1 su questo PC, run2 sul PC di un collega). I risultati sono in `ANALISI_TRAINING/ANALISI_20_05/`.

---

## 2. La scoperta: varianza enorme tra run identiche

Le due run, **a parità di tutto**, hanno prodotto esiti molto diversi. Questo è il dato che ha riorientato l'intera strategia.

### Training (crash rate)

| Metrica | run1 | run2 |
|---|---|---|
| Crash globale | 86.5% | 83.5% |
| Crash M1 | 94.4% | 82.1% |
| M1 spawn P1 (−2.9,−2.0) max-steps | 11.3% | 36.2% |
| M1 spawn P2 (1.0,−1.0) max-steps | 0.0% | 0.0% |
| Crash M2 | 82.0% | 84.3% |

### Test (success rate, 90 episodi × maze)

| Maze | run1 | run2 | Media |
|---|---|---|---|
| M1 | **0%** | **50%** | 25% |
| M2 | 68% | 51% | 59.5% |
| M3 (zero-shot, mai visto) | **0%** | **23%** | 11.5% |

**Lo scarto è enorme:** M1 va da 0% a 50%, M3 da 0% a 23%, M2 da 68% a 51%. Stesso codice. Questo non è "una run buona e una cattiva": è il segnale che **non stiamo misurando una proprietà del modello, ma rumore.**

> Nota positiva da non perdere: il **23% su M3 in run2** è il primo segnale di generalizzazione zero-shot mai prodotto dal progetto. Va protetto, non sovra-ottimizzato.

---

## 3. La causa: nessun seed fissato

Cercando nel codice (`grep` su `seed`, `manual_seed`, `np.random.seed`, `random.seed`, `torch.manual_seed`) **non c'è alcun controllo del seed**. La randomness arriva da sorgenti non controllate:

- inizializzazione pesi della rete (PyTorch),
- campionamento dal replay buffer,
- ε-greedy (azioni esplorative),
- `numpy` e `random` di Python,
- timing non deterministico di Gazebo.

In più, le due run girano su **PC diversi** → si aggiunge un confound hardware *sopra* la varianza da seed. Quindi non possiamo nemmeno separare "varianza da seed" da "differenza tra macchine".

---

## 4. Le conseguenze metodologiche

Questa è la parte scomoda ma importante.

1. **Tutte le conclusioni dei round precedenti erano su n=1.** Ogni "miglioramento" attribuito a una modifica (incluso il famoso "record" R1 M2=82%) era una singola run. Con n=1 non si può attribuire un delta a una modifica: potrebbe essere interamente rumore.

2. **Non possiamo affermare che R-α abbia migliorato o peggiorato.** 82% (R1) vs 68%/51% (R2) potrebbe stare tutto dentro la banda di rumore.

3. **Serve un cambio di metodo prima di qualsiasi confronto futuro:**
   - aggiungere **seed control** (`random.seed` + `np.random.seed` + `torch.manual_seed`, con `--seed` da CLI);
   - girare **≥3–5 seed** della stessa config sullo **stesso PC**;
   - riportare **media ± deviazione standard** (o IQM, Inter-Quartile Mean), **mai il massimo**.

   Gazebo resta non deterministico, quindi il seed non dà riproducibilità bit-a-bit, ma rende la varianza **attribuibile e misurabile** invece che invisibile.

Questo non è un problema solo nostro: è il problema metodologico centrale del deep RL (vedi §6, Henderson 2018 e Agarwal 2021).

---

## 5. Diagnosi dei fallimenti: due classi distinte

Osservando i video del robot nei labirinti e incrociando con i dati per-spawn, i crash si dividono in **due classi che NON vanno confuse**, perché richiedono fix opposti.

### Classe A — Cinematica (problema di controllabilità, NON di RL)

**Unico caso confermato: spawn F1 (−4.5,−3.5).**

Il robot è un modello *unicycle* con velocità lineare fissa. Il raggio minimo di sterzata è:

```
R_min = v / ω_max = 0.5 / 0.8 = 0.625 m
```

Una manovra ad angolo stretto richiede un raggio **< R_min**, ma con velocità fissa **non puoi stringere l'arco** → l'agente fa overshoot e finisce nel muro. Questo crash è **inevitabile per qualunque policy o reward**: non è un problema di apprendimento, è fisica. Un'inversione a U richiederebbe un corridoio largo ≥ 1.25 m.

**Unico fix possibile:** rendere la velocità riducibile. A v=0.25 m/s → R = 0.31 m, e l'agente entra ovunque. Da qui l'idea dell'azione di decelerazione/stop.

### Classe B — Percezione / Reward (problema apprendibile)

**Caso emblematico: spawn M1 P2 (1.0,−1.0), costantemente 0%.** Confermato dall'osservazione: il robot *ce la farebbe* se solo "capisse i dati" — geometricamente la strada libera c'è. Perché allora un value-agent non impara la cosa banale "vai dove non vedi raggi"? Quattro cause sovrapposte:

1. **Il reward dice "non morire", non "vai verso l'aperto".**
   - `space_bonus = 2·mean(scan)/5` usa la **media** di tutti i raggi → è un premio **non direzionale**: non dice *da che parte* andare.
   - `steering_penalty = 0.02·|a−5|` **punisce** la sterzata correttiva (andare dritto, a=5, costa 0).
   - Risultato: miopicamente conviene andare dritto. L'unico segnale per girare è il −1000 bootstrappato ~30 step dopo → **credit assignment** lento e raro, peggiorato da replay uniforme e γ=0.99.

2. **L'MLP è cieco alla struttura spaziale.** La rete 50→300→300→11 è *permutation-blind*: non ha alcun prior che "bin adiacenti = direzioni adiacenti". Il concetto di "lato aperto" è una correlazione da estrarre faticosamente, non un dato strutturale. Una **CNN** lo avrebbe quasi gratis.

3. **Il min-pool maschera l'apertura.** Ogni bin tiene il punto **più vicino** del suo settore (`np.min` per chunk). Un corridoio libero accanto a un muro vicino, nello stesso bin, viene letto come muro. La strada libera può essere **invisibile** nello stato.

4. **Lo yaw assoluto confonde.** `heading = [cos(yaw), sin(yaw)]` è l'orientamento nel mondo, **senza alcun goal**. È un input non causale: confonde e fa overfitting per-maze (probabilmente uccide la generalizzazione su M3).

**Ranking delle cause (priorità):** reward non-direzionale + credit assignment > min-pool masking > MLP senza prior > yaw assoluto.

> Per distinguere con certezza le due classi serve il **logging dell'azione per-step**: a P2 il robot sceglie "dritto" (→ problema reward/credit) o "verso il muro" (→ problema percezione/aliasing)? Sono due fix opposti, non si indovina.

---

## 6. La letteratura e come si applica a noi

I paper qui sotto sono presi dalla nostra reading list (`letteratura_progetto_DRL_collision_avoidance.md`) e cercati/letti online. Per ciascuno: cosa dice e perché ci riguarda.

### 6.1 Riproducibilità — il cuore della Parte 2

- **Henderson et al., 2018 — *Deep Reinforcement Learning that Matters* (AAAI).**
  Dimostra che **stesso algoritmo, stessi iperparametri, seed diversi → risultati completamente diversi**. Distingue sorgenti di varianza estrinseche (iperparametri, codebase) e intrinseche (seed, ambiente). Raccomanda significance testing e reporting su più trial.
  → **È esattamente quello che ci è successo.** La nostra divergenza 0%↔50% è il fenomeno descritto da Henderson, riprodotto involontariamente. Questo paper è la cornice teorica della nostra scoperta.

- **Agarwal et al., 2021 — *Deep RL at the Edge of the Statistical Precipice* (NeurIPS).**
  Propone metriche statistiche corrette per confrontare algoritmi DRL: **IQM**, optimality gap, intervalli di confidenza via bootstrap. I point-estimate da poche run sono inaffidabili.
  → Ci dice **come** riportare i risultati: distribuzione su ≥3–5 seed, IQM/mean±std, mai il max.

### 6.2 Generalizzazione — perché M3 è dura

- **Cobbe et al., 2019 — *Quantifying Generalization in RL* (CoinRun, ICML).**
  Gli agent **overfittano anche con training set grandi**. CNN più profonde + regolarizzazione (L2, dropout, data augmentation, batch norm) migliorano la generalizzazione.
  → Conferma che l'MLP è una scelta debole per generalizzare a M3, e che il multi-maze (più ambienti in training) è una regolarizzazione corretta ma non sufficiente.

- **Kirk et al., 2023 — *A Survey of Zero-shot Generalisation in Deep RL* (JAIR).**
  Formalizza la generalizzazione con i Contextual MDP.
  → Inquadra il limite: M3 zero-shot è un problema di generalizzazione, non di tuning.

- **Zhang et al., 2018 / Dhiman et al., 2018 / "Random Spawns" paper.**
  Random start su mappa fissa **non** previene overfitting; una policy addestrata su N mappe **non** trasferisce automaticamente.
  → Spiega perché spawn diversi su M1/M2 non bastano a far funzionare M3.

### 6.3 Il paper originale e la famiglia Q-learning

- **Van Hasselt et al., 2016 — DDQN (AAAI).** Cuore algoritmico del progetto: risolve l'overestimation di DQN.
- **Feng et al., 2021.** Il paper che riproduciamo: DDQN su LiDAR (512 raggi/270° → 50 bin), azioni angolari discrete, reward semplice (+5/−1000), Gazebo. Confronta DQN / DDQN / DDQN+PER. Ammette esplicitamente un **gap sim-to-real** significativo nelle conclusioni.
  → È la nostra Parte 1. La sua reward semplice e l'azione discreta sono proprio i punti deboli che diagnostichiamo.

### 6.4 Lo stato dell'arte — la Parte 3

- **Confronto diretto 2025 — *Adaptive Emergency Response DRL Navigation* (TurtleBot3, ROS2/Gazebo).** Numeri concreti:

  | Algoritmo | Success | Collisioni | Azione |
  |---|---|---|---|
  | **TD3** | **92%** | 5% | continua |
  | DDPG | 88% | 9% | continua |
  | DQN | 79% | 16% | discreta |

  TD3 riduce il tempo di percorrenza del **28–55%** rispetto a DQN. Setup quasi identico al nostro (ROS2, Gazebo, LiDAR, 8000 episodi, γ=0.99).
  → Evidenza forte e diretta che il passaggio a TD3 con azione continua paga, sullo *stesso tipo* di problema.

- **Fujimoto et al., 2018 — TD3 (ICML).** "Il DDQN degli actor-critic": risolve l'overestimation anche in TD, azione continua. Usato in quasi tutti i paper recenti di navigation.
- **Haarnoja et al., 2018 — SAC (ICML).** Standard de facto per azioni continue: sample-efficient e stabile (entropy-regularized).
- **Schulman et al., 2017 — PPO.** Robusto, supporta discreto e continuo; molto usato come baseline.

### 6.5 Punti trasversali dalla reading list

- **Tobin et al., 2017 — Domain Randomization.** Per il gap sim-to-real (futuro).
- **Reward shaping (§6b reading list).** I paper post-2022 usano reward più ricchi (distanza al goal, proximity penalty progressiva) → convergenza più rapida. Motiva la Parte 1 (la nostra reward è povera) e indica il fix.
- **Azione discreta vs continua (§6c reading list).** DDQN richiede azioni discrete; SAC/TD3 operano in continuo → comportamento più fluido, meno jerk, e — punto chiave per noi — **possono rallentare**, risolvendo la Classe A (F1). È il motivo principale per cui la community si è spostata.

---

## 7. Le domande di inquadramento e le istruzioni dei prof

Prima di proporre una direzione, è stata posta la domanda: **qual è il vincolo dominante del progetto?** (fedeltà al paper / performance massima / rigore metodologico / scadenza).

**Risposta / istruzioni dei prof:** idealmente implementare il paper Feng e, se possibile, proporre miglioramenti. **Ma** i prof hanno detto esplicitamente: *"se vedete che qualcosa non funziona, motivate perché e proponete una soluzione."* Quindi il paper **non è un vincolo rigido** — e a questo punto del lavoro ci siamo comunque già slegati abbastanza da esso.

**Implicazione:** abbiamo mandato libero per trasformare i fallimenti in contributo, purché *motivati*. È esattamente la condizione che rende sensato lo schema a 3 parti.

---

## 8. La decisione strategica: il paper a 3 parti

La struttura narrativa proposta:

> **Parte 1** — DDQN e i suoi limiti
> **Parte 2** — Riproducibilità in DRL (la varianza)
> **Parte 3** — Algoritmi moderni per lo stesso task

Questa non è un ripiego: è il modo corretto in cui un buon paper empirico DRL si struttura quando il metodo originale sotto-rende. Però vanno rispettati due accorgimenti, altrimenti la tesi è attaccabile.

### 8.1 Riframing obbligatorio della Parte 1 — NON dire "DDQN fallisce"

I fallimenti diagnosticati **non sono algoritmici** (non è DDQN-vs-DQN il problema). Sono: reward non direzionale, encoding di stato cieco (min-pool, MLP, yaw), e infeasibilità cinematica a velocità fissa. Se scriviamo "DDQN fallisce", la critica immediata è "non l'avete tunato bene".

La tesi **difendibile** è: *"una riproduzione fedele di DDQN stile-Feng, anche con reward raffinato, si ferma a un success rate basso con varianza enorme su geometrie non viste — ed ecco perché, causa per causa."* Le cause diventano il ponte verso la Parte 3.

### 8.2 Il filo d'oro che lega le tre parti

> **Le limitazioni che troviamo in DDQN sono esattamente quelle che gli algoritmi moderni sono stati progettati per risolvere.**

- L'**azione continua** (TD3/SAC) risolve *gratis* il problema cinematico F1: l'agente può scegliere v=0.1 e stringere la curva.
- Il **reward shaping** moderno risolve la non-direzionalità.
- **Henderson 2018 (Parte 2)** è la colla metodologica: senza seed control, il confronto DDQN-vs-TD3 sarebbe n=1, cioè ripeteremmo lo stesso peccato scoperto adesso, ma al piano superiore.

La Parte 2 (varianza) è il **contributo più originale e più solido**, perché abbiamo prova diretta. Non va sepolta: è ciò che alza il lavoro da "progetto studente" a "lavoro metodologicamente consapevole".

### 8.3 La trappola da evitare: il confronto confuso (confounded)

Se la traccia-DDQN e la traccia-TD3 usano reward / stato / eval diversi, **non si può attribuire nulla** al cambio di algoritmo. Bisogna controllare ciò che è controllabile: **stesso pipeline di stato, stesso set di maze/spawn, stesso protocollo di eval, multi-seed su entrambi.** Altrimenti si rifà l'errore n=1 a livello di algoritmo.

---

## 9. Il piano operativo: spina dorsale + due tracce in parallelo

Siamo in 4 (e abbiamo 4 PC). Le due strade possono girare in parallelo perché condividono l'infrastruttura e differiscono solo nell'agente. Ma c'è una **dipendenza obbligata**: prima si costruisce una base comune, poi si splitta.

### Fase 0 — Spina dorsale (INSIEME, non parallelizzabile) → è la Parte 2

A carico di 1–2 persone, **prima** che chiunque produca numeri comparabili:

- [ ] **Seed control:** `random.seed` + `np.random.seed` + `torch.manual_seed`, con `--seed` da CLI.
- [ ] **Harness di eval unificato:** criterio di valutazione fisso, multi-seed, output **mean ± std / IQM**.
- [ ] **Backup CSV automatico** (`training_log`, `test_results`) in `ANALISI_XX_XX/` prima di ogni reset. *(Lezione già pagata: i CSV grezzi di R1 sono andati persi.)*
- [ ] **Set maze/spawn congelato** e documentato (lo stesso per entrambe le tracce).

Questa fase **è** il contributo metodologico: non è overhead, è la Parte 2 del paper.

### Fase 1 — Split in due tracce (in parallelo)

**Traccia A — DDQN onesto (2 persone) → Parti 1 + 2**
- [ ] Action-space con decelerazione/stop (fixa la Classe A / F1). *Decisione aperta: velocità discrete extra (es. angolare × {0.25, 0.5} = 22 azioni) vs azione "stop" dedicata (11+1).*
- [ ] Logging azione per-step (rende interpretabile P2: "dritto" vs "muro").
- [ ] Multi-seed (≥3–5) → baseline DDQN con **bande di varianza vere**.
- [ ] (Candidato successivo, una variabile alla volta) reward direzionale potential-based (Ng 1999) **se** il log mostra che a P2 sceglie "dritto".

**Traccia B — Algoritmo moderno (2 persone) → Parte 3**
- [ ] Implementare **TD3** sullo **stesso env / stesso eval** (azione continua: risolve F1 gratis; nel confronto 2025 TD3 92% vs DQN 79%).
- [ ] Reward shaping coerente (direzionale) ma il più possibile comparabile con la Traccia A.
- [ ] Multi-seed (≥3–5).
- [ ] (Opzionale, se avanza tempo) SAC come secondo punto di confronto.

### Uso dei 4 PC

Oltre a dividere le due tracce, i 4 PC servono soprattutto a **parallelizzare i seed**: Henderson/Agarwal chiedono ≥3–5 seed per config. Far girare seed diversi su macchine diverse **in parallelo** è dove il gruppo da 4 paga davvero il tempo.

### In una frase

> Continuare il paper *quel tanto che basta* per documentare onestamente perché DDQN-fedele si ferma (Parti 1+2), poi usare quella diagnosi come trampolino motivato verso TD3 (Parte 3) — con una spina dorsale di riproducibilità condivisa che rende il confronto l'unica cosa che conta legittima.

---

## 10. Cosa NON fare

- **Non** girare altri seed dell'action-space attuale: è cinematicamente rotto a F1, sprecheremmo PC-ore.
- **Non** fare un altro round di reward-tuning fine "alla cieca": crash 65–86% sui maze di *training* segnala un problema basilare, non un raffinamento.
- **Non** cambiare algoritmo *dentro* un round senza isolare la variabile (sarebbe scope creep e confonderebbe l'attribuzione).
- **Non** confrontare DDQN e TD3 con reward/stato/eval diversi (confounding → si torna a n=1).
- **Non** riportare mai il massimo di poche run come se fosse il risultato: solo media ± std / IQM.
- **Non** resettare i checkpoint senza prima il backup dei CSV.

---

## 11. Riferimenti bibliografici

**Riproducibilità**
- Henderson et al., 2018 — *Deep Reinforcement Learning that Matters* — AAAI — arXiv:1709.06560
- Agarwal et al., 2021 — *Deep RL at the Edge of the Statistical Precipice* — NeurIPS — arXiv:2108.13264
- Islam et al., 2017 — *Reproducibility of Benchmarked DRL Tasks for Continuous Control*

**Generalizzazione**
- Cobbe et al., 2019 — *Quantifying Generalization in RL (CoinRun)* — ICML — arXiv:1812.02341
- Kirk et al., 2023 — *A Survey of Zero-shot Generalisation in Deep RL* — JAIR — arXiv:2111.09794
- Zhang et al., 2018 — *Study on Overfitting in DRL* — arXiv:1804.06893
- Dhiman et al., 2018 — *A Critical Investigation of Deep RL for Navigation* — arXiv:1802.02274

**Famiglia Q-learning / paper originale**
- Mnih et al., 2015 — *Human-level control through deep RL* — Nature 518
- Van Hasselt et al., 2016 — *Deep RL with Double Q-learning* — AAAI — arXiv:1509.06461
- Schaul et al., 2016 — *Prioritized Experience Replay* — ICLR — arXiv:1511.05952
- Wang et al., 2016 — *Dueling Network Architectures* — ICML — arXiv:1511.06581
- Feng et al., 2021 — DDQN LiDAR navigation (paper riprodotto)

**Stato dell'arte (azione continua)**
- Fujimoto et al., 2018 — *TD3* — ICML — arXiv:1802.09477
- Haarnoja et al., 2018 — *Soft Actor-Critic (SAC)* — ICML — arXiv:1801.01290
- Schulman et al., 2017 — *PPO* — arXiv:1707.06347
- Lillicrap et al., 2015 — *DDPG* — ICLR — arXiv:1509.02971
- *Adaptive Emergency Response DRL Navigation*, 2025 — DQN vs DDPG vs TD3 su TurtleBot3/ROS2/Gazebo — PMC12549260

**Trasversali**
- Tobin et al., 2017 — *Domain Randomization* — IROS
- Tai et al., 2017 — *Virtual-to-real Deep RL for Mapless Navigation* — IROS — arXiv:1703.10945
- Long et al., 2018 — *Decentralized Multi-Robot Collision Avoidance via DRL* — ICRA
- Ng et al., 1999 — *Policy invariance under reward transformations (potential-based shaping)* — ICML

**Libri**
- Sutton & Barto — *Reinforcement Learning: An Introduction* (2nd ed., 2018)
- Goodfellow, Bengio, Courville — *Deep Learning* (2016)

---

*Documento di lavoro. Prossimo passo da decidere col gruppo: scope esatto della Fase 0 (spina dorsale) prima dello split nelle due tracce.*

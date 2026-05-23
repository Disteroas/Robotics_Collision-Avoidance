# Letteratura — DRL Collision Avoidance (Mobile Robot, ROS/Gazebo)

**Progetto:** Riproduzione e aggiornamento di Feng et al. 2021 (DDQN, LiDAR, Gazebo)  
**Obiettivo reading list:** coprire analisi algoritmi, riproducibilità, stato dell'arte attuale

---

## 1. Fondamenta — Algoritmi DRL (famiglia Q-Learning)

Questi sono i paper *fondazionali* che devi conoscere per capire e giustificare le scelte algoritmiche.

### 1.1 DQN originale
- **Mnih et al., 2015** — *Human-level control through deep reinforcement learning*  
  Nature, Vol. 518. DOI: 10.1038/nature14236  
  → Introduce DQN: replay buffer + target network. Base di tutto.

### 1.2 DDQN (il paper che implementi)
- **Van Hasselt, Guez, Silver, 2016** — *Deep Reinforcement Learning with Double Q-learning*  
  AAAI 2016. arXiv: 1509.06461  
  → Spiega il problema di overestimation di DQN e come DDQN lo risolve.  
  **Obbligatorio** — è il cuore algoritmico del tuo progetto.

### 1.3 Double Q-Learning originale (precursore teorico)
- **Van Hasselt, 2010** — *Double Q-learning*  
  NIPS 2010.  
  → Versione tabellare pre-deep. Utile per capire la motivazione teorica.

### 1.4 Prioritized Experience Replay (PER)
- **Schaul et al., 2016** — *Prioritized Experience Replay*  
  ICLR 2016. arXiv: 1511.05952  
  → Feng et al. 2021 lo confronta con DDQN standard — devi capirlo.

### 1.5 Dueling DQN
- **Wang et al., 2016** — *Dueling Network Architectures for Deep Reinforcement Learning*  
  ICML 2016. arXiv: 1511.06581  
  → Separazione V(s) e A(s,a). Spesso combinato con DDQN nell'architettura corrente.

### 1.6 Rainbow — DQN con tutti i miglioramenti
- **Hessel et al., 2018** — *Rainbow: Combining Improvements in Deep Reinforcement Learning*  
  AAAI 2018. arXiv: 1710.02298  
  → Combina DDQN + PER + Dueling + n-step + distributional + noisy nets.  
  Utile per capire dove si è arrivati con la famiglia Q-learning.

---

## 2. Algoritmi policy-gradient e actor-critic (lo stato dell'arte attuale)

Il campo si è spostato quasi completamente verso questi. Capirli ti permette di fare il confronto con DDQN.

### 2.1 PPO
- **Schulman et al., 2017** — *Proximal Policy Optimization Algorithms*  
  arXiv: 1707.06347  
  → Algoritmo di riferimento per molti paper recenti di navigation. Discreto e continuo.

### 2.2 SAC (Soft Actor-Critic)
- **Haarnoja et al., 2018** — *Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL*  
  ICML 2018. arXiv: 1801.01290  
  → Standard de facto per spazi d'azione continui. Sample-efficient, stabile.

### 2.3 TD3 (Twin Delayed DDPG)
- **Fujimoto et al., 2018** — *Addressing Function Approximation Error in Actor-Critic Methods*  
  ICML 2018. arXiv: 1802.09477  
  → "DDQN per gli actor-critic": risolve overestimation anche in TD. Usato in quasi tutti i paper recenti di navigation.

### 2.4 DDPG (precursore di SAC/TD3)
- **Lillicrap et al., 2015** — *Continuous control with deep reinforcement learning*  
  ICLR 2016. arXiv: 1509.02971  
  → Base concettuale per SAC e TD3.

---

## 3. Riproducibilità in DRL

Questa sezione risponde direttamente al punto 2 della tua richiesta: capire perché i risultati di Feng et al. 2021 potrebbero essere difficili da riprodurre, e come documentare correttamente i tuoi.

### 3.1 Il paper fondamentale sulla riproducibilità in DRL
- **Henderson et al., 2018** — *Deep Reinforcement Learning that Matters*  
  AAAI 2018. arXiv: 1709.06560  
  → **Obbligatorio.** Dimostra che: stesso algoritmo, stessi hyperparametri, seed diversi → risultati completamente diversi. Introduce guidelines per reporting corretto. Il benchmark del problema.

### 3.2 Riproducibilità in continuous control
- **Islam et al., 2017** — *Reproducibility of Benchmarked DRL Tasks for Continuous Control*  
  ICML Reproducibility Workshop 2017.  
  → Companion paper di Henderson. Più focalizzato su policy gradient.

### 3.3 Varianza dovuta al seed e alla rete
- **Agarwal et al., 2021** — *Deep Reinforcement Learning at the Edge of the Statistical Precipice*  
  NeurIPS 2021. arXiv: 2108.13264  
  → Propone metriche statistiche corrette per comparare algoritmi DRL (IQM, optimality gap). Essenziale se fai confronti.

### 3.4 Generalizzazione vs overfitting su singola mappa
- **Zhang et al., 2018** — *Study on Overfitting in DRL* (Dissection of overfitting and generalization)  
  arXiv: 1804.06893  
  → Dimostra che random starts NON prevengono overfitting su mappa fissa.

- **Dhiman et al., 2018** — *A Critical Investigation of Deep RL for Navigation*  
  arXiv: 1802.02274  
  → Test diretto: policy addestrata su N mappe non trasferisce a mappe nuove.

- **Paper nel progetto** — *Random Spawns on a Single Map: Necessary but Insufficient*  
  → Già hai questo nel tuo progetto. Citalo quando parli di generalizzazione.

### 3.5 Quantità di training environment e generalizzazione
- **Cobbe et al., 2019** — *Quantifying Generalization in Reinforcement Learning (CoinRun)*  
  ICML 2019. arXiv: 1812.02341  
  → Quante mappe servono per generalizzare? Risposta empirica.

---

## 4. Stato dell'arte — DRL Collision Avoidance (post-2021)

Cosa è cambiato rispetto a Feng et al. 2021? La risposta breve: si è passati da DDQN (discreto) a SAC/TD3 (continuo), con reward shaping più sofisticato e attenzione crescente a safety constraints.

### 4.1 Paper di riferimento per il dominio (pre-2021, fondazionali)
- **Tai et al., 2017** — *Virtual-to-real Deep RL: Continuous Control of Mobile Robots for Mapless Navigation*  
  IROS 2017. arXiv: 1703.10945  
  → Primo lavoro significativo ASAP su navigation mapless con DRL. Usa A3C su LiDAR. Turtlebot3 in Gazebo.  
  **Base di confronto storica.**

- **Long et al., 2018** — *Towards Optimally Decentralized Multi-Robot Collision Avoidance via DRL*  
  ICRA 2018.  
  → Multi-robot, multi-scenario training. Dimostra che diversità di training env è chiave.

### 4.2 Trend 2022–2025: da discreto a continuo, reward shaping, safety

- **[DDPG/TD3-based] Adaptive mapless navigation improved TD3** (2025)  
  Frontiers in Robotics and AI. DOI: 10.3389/frobt.2025.1625968  
  → Confronto ITD3 vs SAC, PPO, A3C su mapless navigation. ITD3 + curiosity + representation learning.

- **[PPO + CBF] RADAR-BPO** — *Risk-Aware RL with Dynamic Safety Filter* (2025)  
  Sensors 2025, DOI: 10.3390/s25175488. PMC: 12431337  
  → PPO + Control Barrier Function. Safety-constrained navigation in Gazebo ROS. ~90% success rate.

- **[TD3] Path planning improved TD3 in dynamic environment** (2024)  
  Heliyon. DOI: 10.1016/j.heliyon.2024.e32167  
  → TD3 migliorato per obstacle avoidance in ambienti dinamici.

- **[PPO/DDPG] Deep RL with Enhanced PPO for Safe Mobile Robot Navigation** (2024)  
  arXiv: 2405.16266  
  → Confronto DDPG vs PPO enhanced, LiDAR, Gazebo/ROS. Architettura NN e reward raffinati.

- **[SAC-P] DRL-Based UAV Navigation with LiDAR + Depth Camera** (2025)  
  Aerospace 2025, DOI: 10.3390/aerospace12090848  
  → SAC con Prioritized sampling (SAC-P). Multi-sensore. 81.23% success rate.

- **[DQN/TD3/DDPG] Adaptive Emergency Response DRL Navigation** (2025)  
  PMC: 12549260  
  → Confronto diretto DQN vs DDPG vs TD3 su TurtleBot3 in ROS2/Gazebo. **TD3 vince.**

- **[Coulomb Force + PPO] Explainable DRL for motion planning** (2025)  
  PMC: 12900773  
  → Reward shaping con forze coulombiane da LiDAR segmentation. TurtleBot v3, Gazebo.

- **[LiDAR FOV analysis]** — *Impact of LiDAR Configuration on Goal-Based Navigation in DRL* (2023)  
  PMC: 10747335  
  → Analisi sistematica di come il FOV del LiDAR impatta le performance DRL. Rilevante per il tuo setup.

### 4.3 Survey recenti (panoramica sistematica)

- **Paper nel progetto** — *Deep RL for Robotics: A Survey of Real-World Successes*  
  Annual Reviews. Già nel tuo progetto — contiene una sezione specifica su navigation.

- **Paper nel progetto** — *RL in Robotics: A Comprehensive Survey* (Singh et al.)  
  Già nel tuo progetto.

- **Kirk et al., 2023** — *A Survey of Zero-shot Generalisation in Deep RL*  
  JAIR 2023. arXiv: 2111.09794  
  → Formalizza il problema di generalizzazione con Contextual MDPs. Utile per inquadrare il limite di Feng et al. 2021.

- **Zhu & Zhang, 2021** — *DRL Based Mobile Robot Navigation: A Review*  
  Tsinghua Science and Technology.  
  → Review completa DRL + navigation. Buon punto di partenza per inquadrare il campo.

---

## 5. Libri fondazionali

- **Sutton & Barto** — *Reinforcement Learning: An Introduction* (2nd ed., 2018)  
  http://incompleteideas.net/book/the-book-2nd.html — **Free online**  
  → Capitoli 6 (TD), 9 (function approximation), 13 (policy gradient). Basi teoriche di tutto.

- **Goodfellow, Bengio, Courville** — *Deep Learning* (2016)  
  https://www.deeplearningbook.org — **Free online**  
  → Per la parte di architetture neurali. Capitoli 6-9 sufficienti.

---

## 6. Punti che suggerisco di aggiungere alla tua ricerca

Oltre ai 3 punti che hai listato, considera:

### 6a. Sim-to-real gap
Il paper di Feng et al. 2021 ha un gap significativo tra simulazione e real-world (lo ammette nella sezione conclusioni). Questo è un problema aperto enorme.  
**Reference chiave:** Tobin et al., 2017 — *Domain Randomization for Transferring DRL Policies* (IROS 2017).

### 6b. Reward shaping
Feng et al. usa una reward function molto semplice (-1000/+5). Quasi tutti i paper post-2022 usano reward shaping più sofisticato (distanza al goal, velocità angolare, proximity penalty progressiva).  
→ Impatta enormemente la velocità di convergenza. Analizzarlo ti permette di giustificare modifiche al paper originale.

### 6c. Spazio d'azione discreto vs continuo
DDQN richiede azioni discrete (11 velocità angolari nel paper). I metodi moderni (SAC, TD3) operano in spazi continui → comportamento più fluido, meno jerk. Questo è uno dei principali motivi per cui la community si è spostata.  
→ Puoi usarlo come motivazione per confrontare il tuo DDQN con un SAC/TD3 equivalente.

---

## Ordine di lettura consigliato

1. Sutton & Barto cap. 6, 9 (basi)
2. Mnih 2015 (DQN)
3. Van Hasselt 2016 (DDQN) ← cuore del progetto
4. Schaul 2016 (PER) ← usato nel paper
5. Feng et al. 2021 ← il tuo paper
6. Henderson 2018 ← riproducibilità
7. Agarwal 2021 ← come misurare correttamente
8. Zhang 2018 + Random Spawns paper ← generalizzazione
9. Fujimoto 2018 (TD3) + Haarnoja 2018 (SAC) ← stato dell'arte
10. Survey recenti 2024-2025 ← cosa è cambiato

---

*Generato: Maggio 2026*

# Riproducibilità, seed e non-determinismo hardware nel training DRL

**Progetto:** DDQN collision avoidance per UGV (Gazebo / ROS 2 / Docker)
**Branch:** `paper_metric_base` (congelato @ `474a363`)
**Data:** 2026-05-24
**Scopo:** Documentare l'evidenza empirica che il controllo del seed è *necessario ma non sufficiente* per la riproducibilità quando il simulatore è non-deterministico, definire il protocollo sperimentale multi-seed / multi-macchina adottato, e fornire l'ancoraggio alla letteratura per la Parte 2 del paper (riproducibilità).

---

## 1. Sintesi (TL;DR)

Due training con **seed identico** (`seed=1`), **codice identico** (stesso `git_sha` `474a363`) e **configurazione identica** (`r_alpha`), eseguiti su **due macchine diverse** (GrindMachine vs MSI), producono **policy finali drasticamente diverse**: il maze di test zero-shot M3 passa da **100%** a **3.3%** di success rate.

Il seed fissa il PRNG (init pesi, sampling azioni ε-greedy, sampling replay, scelta spawn) ma **non** fissa la dinamica del simulatore Gazebo, che è accoppiata al wall-clock e quindi dipendente dall'hardware. La piccola perturbazione fisica iniziale viene amplificata in due modi (desincronizzazione dello stream RNG + bootstrapping caotico del DDQN) fino a produrre policy qualitativamente diverse dopo 5000 episodi.

**Conseguenza metodologica:** il seed va trattato come variabile di disturbo da marginalizzare (mediare su ≥5 seed, riportare distribuzione/IQM/CI), **non** come iperparametro da scegliere. La macchina è un confondente: i seed prodotti su hardware diverso **non** sono campioni i.i.d. dello stesso processo e non vanno aggregati insieme.

---

## 2. Setup del confronto controllato

Provenienza letta da `run_meta.json` delle due run:

| Campo | GrindMachine (locale) | MSI (collega) |
|---|---|---|
| `config` | r_alpha | r_alpha |
| `seed` | 1 | 1 |
| `git_sha` | 474a363… | 474a363… |
| `hostname` | GrindMachine | MSI |
| `success_criterion` | MAX_STEPS=500 senza collisione, ε=0.0 | idem |
| `timestamp` | 2026-05-23T20:03 | 2026-05-24T08:53 |

È il confronto controllato ideale: **una sola variabile cambia (la macchina)**. Se il training fosse deterministico dato il seed, i risultati sarebbero identici. Non lo sono.

> **Avvertenza sul termine "hardware".** `run_meta.json` registra solo `git_sha` + `hostname`, non le versioni di PyTorch / Gazebo / ROS / la data dell'ultimo `colcon build`. Quindi la variabile che cambia è "**la macchina**" intesa come stack completo (CPU + OS + librerie + build), non la CPU isolata. La divergenza è dimostrata; l'attribuzione *esclusiva* alla CPU non lo è. Vedi §7 (limiti).

---

## 3. Risultati: divergenza in valutazione

Test greedy (ε=0.0), copertura round-robin degli spawn, da `eval_summary.csv`.

| Maze | GrindMachine | MSI (Gio) | Δ |
|---|---|---|---|
| M1 (train) | **23.3%** (14/60) | **0%** (0/60) | −23.3 pp |
| M2 (train) | **37.8%** (68/180) | **18.9%** (34/180) | −18.9 pp |
| M3 (test zero-shot) | **100%** (30/30) | **3.3%** (1/30) | **−96.7 pp** |

Dettaglio reward/step medi:

| Maze | Grind avg_reward / avg_steps | MSI avg_reward / avg_steps |
|---|---|---|
| M1 | 104.5 / 285.2 | 318.4 / 301.6 |
| M2 | 1544.9 / 426.0 | 994.2 / 363.5 |
| M3 | 2665.1 / 500.0 | −243.4 / 254.5 |

Stesso seed → policy completamente diverse. Il collasso di M3 (100% → 3.3%) è di gran lunga maggiore della varianza inter-seed osservata finora sul progetto: **la varianza-macchina qui domina la varianza-seed.**

---

## 4. La divergenza parte dall'episodio 1

Episodio 1 — **stesso spawn** `(-2.9, -2.0)` (primo draw RNG identico, come atteso dal seed condiviso):

| | GrindMachine | MSI |
|---|---|---|
| step | 218 | 217 |
| reward | 0.57 | −0.28 |

Già diverso. E all'episodio 2 la **sequenza degli spawn diverge** (Grind resta su `(-2.9,-2.0)`, MSI passa a `(1.0,-1.0)`). Se il PRNG fosse l'unica sorgente di casualità, le sequenze di spawn sarebbero identiche su entrambe le macchine. Non lo sono → c'è una sorgente di divergenza **fuori dal PRNG**.

---

## 5. Traiettoria di training: identica all'inizio, caotica alla fine

Da `training_log.csv`, milestone `avg100` / `total_crashes`:

| ep | Grind (avg100 / crashTot) | MSI (avg100 / crashTot) |
|---|---|---|
| 1000 | −138 / **1000** | −188 / **1000** |
| 2000 | −316 / 1995 | −312 / 1992 |
| 3000 | 954 / 2923 | 948 / 2859 |
| 4000 | 1358 / 3785 | 1503 / 3650 |
| 5000 | 94 / 4538 | 352 / 4371 |

Fino a ~ep1000 le due run sono **quasi identiche** (crash totali 1000 = 1000; a ep2000 differiscono di 3). La divergenza inizia a contare intorno a ep3000 e **accelera** dopo. È la firma di un sistema **caotico**: differenze micro accumulate fino a divergenza macroscopica (butterfly effect). L'inizio è dominato dall'esplorazione random (ε≈1, azioni seedate → quasi uguali); la divergenza emerge quando la policy *appresa* (che dipende dal contenuto accumulato nel replay) prende il sopravvento col decadere di ε.

---

## 6. Meccanismo: perché il seed non basta

Il seed **non** è la causa della divergenza. Il seed fissa: init dei pesi (draw Kaiming/Xavier), ordine di sampling delle azioni ε-greedy, indici di sampling del replay, `random.choice` degli spawn. La causa è il **simulatore**, attraverso tre stadi:

### 6.1 Gazebo non-deterministico (sorgente)
Il loop di step (`_wait_sim_seconds(0.1)`, spin cycles, drain di 20 cicli LIDAR dopo il reset) è accoppiato al wall-clock. Su CPU diverse cambiano il numero di substep di integrazione fisica per azione, il timing di arrivo dei messaggi e gli arrotondamenti in virgola mobile (non-associatività, SIMD/BLAS diversi, multithreading). CLAUDE.md lo dichiara esplicitamente: *"Gazebo physics/timing stays non-deterministic, so there is no bit-for-bit reproducibility."* Questo è coerente con Nagarajan et al. (2018), che cataloga le sorgenti di non-determinismo nel DRL.

### 6.2 Desincronizzazione dello stream RNG (amplificatore)
Il loop spawn-retry in `usv_env.py` chiama `random.choice(spawn_list)` e poi verifica la clearance LIDAR; se la verifica fallisce ritenta. Se il LIDAR differisce (per §6.1), il loop itera un numero **diverso** di volte tra le due macchine → consuma un numero **diverso** di draw dal PRNG → lo **stream RNG globale si desincronizza**. Dopo la desincronizzazione, anche le componenti "controllate dal seed" (azioni ε-greedy, sampling del replay) divergono. Conferma empirica: la sequenza spawn diverge già da ep2 (§4). Una perturbazione fisica infinitesima si trasforma così in divergenza completa dello stream pseudo-casuale.

### 6.3 Amplificazione per bootstrapping (esplosione)
Il DDQN aggiorna `Q(s,a) ← r + γ·max_a' Q_target(s',a')`. Il replay buffer accumula transizioni divergenti; la target network copia pesi divergenti ogni 1000 step. Le differenze precoci si propagano e si amplificano in modo moltiplicativo lungo 5000 episodi. Il feedback loop dati→policy→dati, tipico dell'RL, rende il problema **più severo** che nel supervised learning (Henderson 2018, Islam 2017).

### 6.4 Il paradosso: training "migliore" ≠ generalizzazione migliore
MSI ha trainato **meglio** sui numeri grezzi — sopravvivenza in training 12.6% vs 9.2%, reward medio 400 vs 272, M2 train-survive 17.8% vs 10.4% — ma **generalizza peggio** (M3 3.3% vs 100%). Lezione: con due run divergenti **non si possono nemmeno ordinare le macchine usando le metriche di training**. La selezione del best-model su `avg100` globale può premiare una policy che sopravvive di più sugli spawn di training (divergenti) ma è fragile in transfer.

---

## 7. La domanda teorica: i seed sono "solo numeri"? Esistono seed migliori?

**Il numero del seed in sé è irrilevante.** seed 0 ≡ 42 ≡ 3407. Nessuna proprietà aritmetica; la mappa seed→performance è caotica e non-liscia (seed 41/42/43 non correlano), quindi non esiste "ricerca vicino a un buon seed".

**Esistono seed migliori? In due sensi distinti:**

1. **Post-hoc, sì (empiricamente).** Seed diversi → performance finali diverse, a volte con bande non sovrapposte (Henderson 2018). Ma è una proprietà *dell'interazione* tra quello stream RNG e il landscape di ottimizzazione, **non** del numero.
2. **Intrinsecamente / trasferibilmente, no.** Un seed "buono" è specifico della pipeline (cambia 1 neurone → il seed buono diventa arbitrario) e — come dimostrato qui — non è nemmeno stabile cross-machine col nostro non-determinismo.

**Perché alcuni seed sembrano migliori (meccanismi reali):**
- **Lottery Ticket Hypothesis** (Frankle & Carbin 2019): l'init random contiene "sottoreti vincenti"; alcuni init sono meglio posizionati per SGD. Effetto reale, ma è un draw fortunato della distribuzione di init, non un numero speciale.
- **Traiettoria di esplorazione (RL-specifico):** le prime azioni random riempiono il replay; dati precoci fortunati → bootstrapping migliore. Feedback loop dati↔policy → RL più sensibile al seed del supervised.
- **Non-convessità:** init diversi → bacini di attrazione diversi → generalizzazione diversa.

**Il seed NON è un iperparametro.** Un iperparametro si "spedisce" (es. learning rate); non si spedisce "usa seed 1". Un metodo deve funzionare *across* seed. Il seed è un campione del rumore da **marginalizzare** per stimare la performance attesa dell'algoritmo. Riportare il *best seed* è p-hacking (Henderson 2018 mostra che splittando 5 seed in due gruppi si possono "dimostrare" differenze tra algoritmi identici; Picard 2021 quantifica lo spread e il rischio di "seed hacking").

**Nuance specifica al progetto.** Col non-determinismo Gazebo, da noi il seed è un controllo *ancora più debole* del benchmark RL standard: varianza-seed e stocasticità-ambiente sono **confuse**, e non riproduciamo "seed 1" bit-for-bit nemmeno sulla stessa macchina. I nostri "5 seed" sono quindi più vicini a **"5 run stocastiche indipendenti che condividono l'init"** — il che è onesto per stimare la varianza reale, ma va inquadrato così nel paper, non come repliche perfettamente controllate.

---

## 8. Limiti di questa analisi

- **n = 1 per macchina.** La divergenza è dimostrata in modo airtight (stesso seed → esiti diversi ⇒ esiste una sorgente non-PRNG). Ma un bias *sistematico* ("MSI è sistematicamente peggio") richiede ≥3 seed per macchina: con n=1 non distinguiamo "MSI peggiore in media" da "seed=1 particolarmente sensibile".
- **"Macchina" = stack completo**, non CPU isolata (§2). Per isolare servirebbe loggare versioni di torch/Gazebo/ROS e data di build.
- **Rumore di valutazione non isolato.** I due `best_model.pth` sono diversi → la differenza in test mischia divergenza-training + rumore-eval-Gazebo. Esperimento netto per separarli (rimandato): valutare `PC_GIO/seed_1/best_model.pth` su GrindMachine. Se M3 resta ~3% → policy genuinamente peggiore; se risale ~100% → era rumore eval-machine.

---

## 9. Protocollo sperimentale adottato

Decisione operativa (2026-05-24): **5 seed su GrindMachine + 5 seed su MSI, in parallelo**, per ottenere due distribuzioni within-machine e quantificare al contempo l'effetto macchina.

Regole (derivate dalla letteratura in §10):
1. **Seed fissati a priori e documentati**: `0,1,2,3,4`. Non scelti, non cambiati dopo aver visto i risultati. Stesso set su entrambe le macchine → confronto **paired** per-seed (massima potenza statistica).
2. **Due distribuzioni separate** `{Grind: 5 seed}` e `{MSI: 5 seed}`. **Non** aggregare i 10 come unica distribuzione (confondente hardware). Una macchina è la **gold-standard headline**; l'altra serve a misurare la riproducibilità cross-machine.
3. **Mai riportare il best seed.** Si riporta la distribuzione: **IQM + 95% bootstrap CI** (Agarwal 2021) — già implementato in `aggregate_seeds.py` — oppure mean±std (Henderson 2018, Colas 2018).
4. **Nessun tuning sui seed che verranno riportati** (no leakage / no selezione sul metric di test).
5. **5 seed potrebbero non bastare** data la varianza enorme (100%↔3.3%). Se il CI resta largo, aumentare n (power analysis, Colas 2018). Più seed > analisi più sofisticata.
6. **Migliorare il logging di provenienza**: aggiungere a `run_meta.json` versioni torch/Gazebo/ROS, `OMP_NUM_THREADS`, data `colcon build`, per rendere attribuibile il confondente "macchina".

Mitigazioni opzionali per ridurre (non eliminare) il non-determinismo: `torch.use_deterministic_algorithms(True)`, `OMP_NUM_THREADS=1`, flag CuDNN deterministici. Il timing di Gazebo resta non-deterministico → riproducibilità bit-for-bit non raggiungibile senza un refactor verso stepping fisico in lockstep disaccoppiato dal wall-clock.

---

## 10. Riferimenti bibliografici

**Riproducibilità e varianza nel DRL (nucleo della Parte 2):**

- **Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018).** *Deep Reinforcement Learning that Matters.* AAAI 2018. arXiv:1709.06560.
  → Riferimento principale: seed diversi → risultati statisticamente diversi; rischio di p-hacking sui seed; raccomanda distribuzioni, non single-run.
- **Islam, R., Henderson, P., Gomrokchi, M., Precup, D. (2017).** *Reproducibility of Benchmarked Deep Reinforcement Learning Tasks for Continuous Control.* ICML 2017 Reproducibility Workshop. arXiv:1708.04133.
  → Varianza tra seed e tra codebase nei task di controllo continuo.
- **Colas, C., Sigaud, O., Oudeyer, P-Y. (2018).** *How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.* arXiv:1806.08295.
  → Analisi di potenza statistica: quanti seed servono; uso di bootstrap CI.
- **Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., Bellemare, M.G. (2021).** *Deep Reinforcement Learning at the Edge of the Statistical Precipice.* NeurIPS 2021 (Outstanding Paper). arXiv:2108.13264. Libreria `rliable`.
  → IQM, performance profiles, stratified bootstrap CI; pochi seed ⇒ stime puntuali inaffidabili.
- **Nagarajan, P., Warnell, G., Stone, P. (2018).** *Deterministic Implementations for Reproducibility in Deep Reinforcement Learning.* arXiv:1809.05676.
  → Cataloga le sorgenti di non-determinismo (RNG, GPU/threading, ambiente) e come renderle deterministiche.
- **Pineau, J., et al. (2021).** *Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program).* JMLR 22(164).
  → ML Reproducibility Checklist: riportare seed, hardware, iperparametri.
- **Machado, M.C., Bellemare, M.G., Talvitie, E., Veness, J., Hausknecht, M., Bowling, M. (2018).** *Revisiting the Arcade Learning Environment: Evaluation Protocols and Open Problems for General Agents.* JAIR 61. arXiv:1709.06009.
  → Protocolli di valutazione e iniezione di stocasticità (sticky actions) per evitare overfitting al determinismo.

**Inizializzazione / sensibilità al seme casuale:**

- **Frankle, J., Carbin, M. (2019).** *The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks.* ICLR 2019. arXiv:1803.03635.
  → L'init random contiene sottoreti più allenabili: l'init conta.
- **Picard, D. (2021).** *Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision.* arXiv:2109.08203.
  → Dimostrazione empirica dello spread dovuto al solo seed e del rischio di "seed hacking".

**Algoritmo e generalizzazione (contesto progetto):**

- **Mnih, V., et al. (2015).** *Human-level control through deep reinforcement learning.* Nature 518. DOI:10.1038/nature14236. → DQN: replay + target network.
- **Van Hasselt, H., Guez, A., Silver, D. (2016).** *Deep Reinforcement Learning with Double Q-learning.* AAAI 2016. arXiv:1509.06461. → DDQN, cuore algoritmico del progetto.
- **Cobbe, K., Klimov, O., Hesse, C., Kim, T., Schulman, J. (2019).** *Quantifying Generalization in Reinforcement Learning.* ICML 2019. arXiv:1812.02341. → Multi-environment training come regolarizzazione; valutazione su livelli non visti (rilevante per M3 zero-shot).

---

*Documento collegato a: `letteratura_progetto_DRL_collision_avoidance.md`, `Riassunto Deep Reinforcement Learning that Matters.md`, `report_spawn_generalizzazione_DRL.md`. Dati grezzi: `runs/r_alpha/seed_1/` (GrindMachine) e `runs/r_alpha/PC_GIO/seed_1/` (MSI).*

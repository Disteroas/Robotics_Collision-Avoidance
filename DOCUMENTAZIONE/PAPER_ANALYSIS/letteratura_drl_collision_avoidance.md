# Letteratura DRL per collision avoidance — fonti autorevoli, confronto con Feng, allineamento metodo/metrica

**Data:** 2026-05-25 · Documento di analisi interno (IT).
**Scopo:** dare al progetto un'ancoratura bibliografica solida (oltre Feng 2021), capire se le fonti autorevoli usano il **nostro stesso metodo e la nostra stessa metrica**, e collocare Feng rispetto ad esse.

Collegato a: [`riproducibilita_seed_hardware.md`](riproducibilita_seed_hardware.md) (teoria seed + non-determinismo) e al report empirico [`../report_5seed_riproducibilita/`](../report_5seed_riproducibilita/).

---

## 0. Recap di quanto stabilito in questa sessione

1. **Seed.** Non è un iperparametro ma una **variabile di disturbo** da marginalizzare. Cinque run identici per codice/iperparametri, diversi solo per seed, danno esiti molto diversi (Maze 3 bimodale 0/100). Single-run e best-seed sono entrambi fuorvianti (Henderson 2018, Agarwal 2021). Vedi il report a 5 seed.
2. **Feng 2021** (il nostro "paper di riferimento" storico) **non risulta autorevole**: non compare nelle rassegne SOTA né tra i landmark del settore; metrica favorevole ("0 collisioni in 5 minuti") e reporting minimo (probabile single-run). Decisione: **declassarlo** da oracolo a "una delle implementazioni DDQN-USV consultate" e spostare le ancore forti sui landmark qui sotto + sui paper di metodo.
3. **Conseguenza pratica:** questo documento fornisce le ancore alternative e verifica che il nostro protocollo (success-rate, ε=0 greedy, test su maze held-out, distribuzione multi-seed) sia in linea con — anzi più rigoroso di — la pratica del settore applicato.

---

## 1. Le fonti autorevoli — quadro generale

Quattro lavori sono i pilastri del *DRL per navigazione/collision avoidance con sensori range* (LIDAR/laser), il filone esattamente sovrapposto al nostro (ingresso range-finder → comandi di movimento, senza mappa). Sono pubblicati nelle venue top di robotica (ICRA/IROS) e altamente citati.

### 1.1 Tai, Paolo & Liu 2017 — *Virtual-to-real DRL: Continuous Control of Mobile Robots for Mapless Navigation* (IROS 2017, arXiv:1703.00420)
- **Cosa fa:** motion planner *mapless* end-to-end. Input = **10 fasci laser sparsi** + posizione del target nel frame del robot; output = comandi di sterzata **continui**. Addestrato con un metodo DRL asincrono (ADDPG, variante asincrona di DDPG).
- **Perché conta:** è il prototipo del "da scansione laser direttamente all'azione, senza mappa". Dimostra il **transfer sim-to-real**.

### 1.2 Long et al. 2018 — *Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep RL* (ICRA 2018, arXiv:1709.10082)
- **Cosa fa:** policy di collision avoidance **sensor-level decentralizzata** per più robot. Mappa misure laser grezze → velocità di movimento. Addestrata con **PPO** (policy-gradient on-policy), in simulazione massivamente parallela, poi trasferita su robot reali.
- **Estensione:** Fan et al. 2020, *Distributed multi-robot collision avoidance via DRL for navigation in complex scenarios* (IJRR, doi:10.1177/0278364920916531) — versione su rivista, scenari complessi, integrazione con planning.
- **Perché conta:** stato dell'arte sul *sensor-to-action* per evitamento ostacoli; baseline metodologico per il reward shaping e il setup di valutazione a scenari.

### 1.3 Chen, Liu, Everett & How 2017 — *Decentralized Non-communicating Multiagent Collision Avoidance with DRL* (CADRL) (ICRA 2017, arXiv:1609.07845)
- **Cosa fa:** **value network** che stima il tempo-al-goal data la configurazione congiunta (posizioni/velocità) dell'agente e dei vicini. Confrontato con ORCA come baseline.
- **Estensione:** Everett, Chen & How 2018 (GA3C-CADRL, IROS) — più agenti, attori asincroni.
- **Differenza dal nostro:** l'input è **agent-state** (posizioni/velocità dei vicini), non raw LIDAR. Rilevante più per il *framing* del problema multi-agente che per l'architettura percettiva.

### 1.4 Woo & Kim 2020 — *Collision avoidance for an unmanned surface vehicle using DRL* (Ocean Engineering, S0029801820300792)
- **Cosa fa:** il landmark **USV** del filone. DRL per evitamento ostacoli su veicolo di superficie; valutazione su scenari (incl. considerazioni COLREGs in lavori successivi).
- **Perché conta:** è l'ancora di dominio corretta per noi (USV), molto più citata e visibile di Feng.

### 1.5 Contesto / metodo (già citati altrove)
- **Henderson et al. 2018** (arXiv:1709.06560), **Islam et al. 2017** (arXiv:1708.04133), **Agarwal et al. 2021** (rliable, arXiv:2108.13264), **Colas et al. 2018** (arXiv:1806.08295): la letteratura sulla **crisi di riproducibilità** del DRL e su come riportare i risultati (multi-seed, IQM, CI, power analysis).
- **Cobbe et al. 2019** (arXiv:1812.02341): generalizzazione con split **train/test** di ambienti — giustifica il nostro Maze 3 held-out.

---

## 2. Le fonti mirate al NOSTRO obiettivo

Il nostro task in concreto: **LIDAR (50 bin) → MLP → 11 azioni angolari discrete (DDQN)**, navigazione in maze, velocità lineare fissa, valutazione su maze noti (M1, M2) e uno held-out (M3).

| Fonte | Input | Azione | Algoritmo | Vicinanza al nostro setup |
|---|---|---|---|---|
| **Tai 2017** | 10 fasci laser + target | continua | ADDPG | **Alta** sul lato percettivo (laser→azione, mapless). Diversa l'azione (continua vs discreta). |
| **Long 2018 / Fan 2020** | laser grezzo | velocità (continua) | PPO | **Alta** su sensor-to-action e setup di valutazione a scenari; multi-robot (noi single-agent). |
| **Chen/CADRL 2017** | agent-state | — | value net (DQN-like) | Media: utile per il framing, ma input non-LIDAR. |
| **Woo & Kim 2020** | sensori USV | manovra | DRL | **Alta** sul dominio (USV); ancora di riferimento corretta. |
| **Feng 2021** | — | discreta | DQN/DDQN/PER | Vicino sul metodo (DDQN discreto), ma poco autorevole (vedi §3). |

**Lettura mirata.**
- Sul **lato percettivo e di azione**, il parente più stretto è **Tai 2017** (laser→azione mapless): conferma che il paradigma "range-finder → policy" è valido e trasferibile. La differenza chiave è azione continua (DDPG) vs la nostra discreta (DDQN); per azioni discrete il riferimento naturale resta la famiglia DQN/DDQN (Mnih 2015, Van Hasselt 2016).
- Sul **dominio**, l'ancora corretta è **Woo & Kim 2020** (USV), da preferire a Feng come riferimento di settore.
- Sul **rigore di valutazione**, le ancore sono i paper di metodo (Henderson, Agarwal, Colas) — che è proprio ciò che applichiamo nel report a 5 seed e che la maggior parte dei paper applicativi **non** fa.

---

## 3. Confronto: fonti autorevoli vs Feng 2021

| Criterio | Landmark (Tai/Long/Chen/Woo) | Feng 2021 |
|---|---|---|
| **Venue** | ICRA / IROS / Ocean Engineering (top) | minore / poco visibile |
| **Citazioni / visibilità** | alte; compaiono nelle rassegne SOTA | assente dalle rassegne SOTA trovate |
| **Baseline di confronto** | sì (es. CADRL vs ORCA; Long vs metodi classici) | confronto interno DQN/DDQN/PER, no baseline esterno forte |
| **Metrica** | success-rate / collision-rate / tempo-al-goal su scenari standard | "0 collisioni in 5 minuti" — metrica continua **favorevole** (il robot può rallentare) |
| **Reporting statistico** | per lo più **single training run**, valutazione su molti episodi/scenari (norma del settore pre-2018) | single-run, reporting minimo |
| **Transfer / generalizzazione** | spesso sim-to-real o scenari multipli | maps a difficoltà crescente (future work) |

**Sintesi.** Feng è debole **non** solo perché single-run (lo sono anche molti landmark del 2017–18: era la norma), ma per la **combinazione**: bassa autorevolezza + metrica non-standard e favorevole + nessun baseline esterno forte + reporting scarno. I landmark, pur usando spesso un solo run, adottano **metriche standard** (success/collision rate) e **si confrontano con baseline** riconosciute. Per questo restano riferimenti validi mentre Feng va declassato.

---

## 4. Queste fonti rispecchiano il NOSTRO metodo e la NOSTRA metrica?

Distinguiamo **metrica** (cosa misuriamo) e **metodo/rigore** (come lo riportiamo).

### 4.1 Metrica — SÌ, allineata
La nostra metrica primaria è il **success rate** (raggiungere 500 step senza collisione, ε=0 greedy). È esattamente la metrica-standard del settore: success-rate / collision-rate su episodi o scenari (Tai, Long, Woo). Inoltre:
- **ε=0 (greedy) in valutazione**: corretto e standard — si valuta la policy, non l'esplorazione.
- **Maze 3 held-out**: in linea con la pratica di valutare la generalizzazione su ambienti non visti (Cobbe 2019). Molti paper di navigazione testano in scenari nuovi.
- **Round-robin sugli spawn**: rende la copertura bilanciata e onesta (forza nel conteggio anche gli spawn sempre-falliti). Pochi paper lo esplicitano, ma va nella direzione del rigore.

→ **La nostra metrica è quella giusta e condivisa dalla letteratura.**

### 4.2 Metodo / rigore — PIÙ rigoroso della media applicativa
Qui c'è la sfumatura importante:
- La maggior parte dei **paper applicativi** (Tai, Long, Woo, Feng) riporta **un solo training run**, valutato su molti episodi. Riportano statistiche **sugli episodi di valutazione**, ma **non** sulla distribuzione **across-seed**.
- La pratica **multi-seed con IQM/CI** (che noi applichiamo) proviene dalla letteratura di **metodo** (Henderson 2018, Agarwal 2021, Colas 2018), non da quella applicativa.

→ **Il nostro protocollo segue la metrica del settore applicato ma adotta il rigore statistico del settore metodologico** — una combinazione che la maggior parte dei paper applicativi NON ha. Questo è un **punto di forza** difendibile: stiamo misurando le cose giuste (come il settore) e le riportiamo come la best-practice metodologica richiede (cosa che il settore applicato spesso omette). L'unico limite onesto è la **numerosità** (n=5, sotto-potenza): vedi il report a 5 seed, §6.

### 4.3 Dove ci discostiamo (consapevolmente)
- **Azione discreta (DDQN)** vs continua (DDPG/PPO) di Tai/Long: scelta legittima per uno spazio d'azione piccolo; riferimento DQN/DDQN (Mnih 2015, Van Hasselt 2016).
- **Single-agent** vs multi-robot di Long/Chen: il nostro problema è un veicolo singolo in ambiente statico, quindi il framing multi-agente non si applica direttamente.

---

## 5. Raccomandazioni operative

1. Nella relazione finale, **citare Woo & Kim 2020** come riferimento di dominio USV e **Tai 2017 / Long 2018** come riferimenti del paradigma laser→azione; **Henderson 2018 + Agarwal 2021** per giustificare il protocollo multi-seed.
2. **Declassare Feng** a riferimento secondario, segnalandone esplicitamente i limiti di metrica e reporting.
3. Vendere come **punto di forza** il fatto che adottiamo metrica standard + rigore multi-seed (IQM/CI), esplicitando il limite di numerosità (n=5).
4. Per i lettori che chiedono "perché non riproducete il 100% di Feng": rispondere con il report a 5 seed (la bimodalità 0/100 dimostra empiricamente perché un singolo "0 collisioni" non è affidabile).

---

## 6. Riferimenti

- **Tai, L., Paolo, G., Liu, M. (2017).** Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation. *IROS 2017.* arXiv:1703.00420.
- **Long, P., Fan, T., Liao, X., Liu, W., Zhang, H., Pan, J. (2018).** Towards Optimally Decentralized Multi-Robot Collision Avoidance via Deep Reinforcement Learning. *ICRA 2018.* arXiv:1709.10082.
- **Fan, T., Long, P., Liu, W., Pan, J. (2020).** Distributed multi-robot collision avoidance via deep reinforcement learning for navigation in complex scenarios. *IJRR.* doi:10.1177/0278364920916531.
- **Chen, Y. F., Liu, M., Everett, M., How, J. P. (2017).** Decentralized Non-communicating Multiagent Collision Avoidance with Deep Reinforcement Learning (CADRL). *ICRA 2017.* arXiv:1609.07845.
- **Everett, M., Chen, Y. F., How, J. P. (2018).** Motion Planning Among Dynamic, Decision-Making Agents with Deep Reinforcement Learning (GA3C-CADRL). *IROS 2018.*
- **Woo, J., Kim, N. (2020).** Collision avoidance for an unmanned surface vehicle using deep reinforcement learning. *Ocean Engineering.* (ScienceDirect S0029801820300792).
- **Henderson, P. et al. (2018).** Deep Reinforcement Learning that Matters. *AAAI.* arXiv:1709.06560.
- **Islam, R. et al. (2017).** Reproducibility of Benchmarked Deep RL Tasks for Continuous Control. arXiv:1708.04133.
- **Agarwal, R. et al. (2021).** Deep RL at the Edge of the Statistical Precipice (rliable). *NeurIPS.* arXiv:2108.13264.
- **Colas, C. et al. (2018).** How Many Random Seeds? Statistical Power Analysis in DRL. arXiv:1806.08295.
- **Cobbe, K. et al. (2019).** Quantifying Generalization in Reinforcement Learning. *ICML.* arXiv:1812.02341.
- **Mnih, V. et al. (2015).** Human-level control through deep reinforcement learning. *Nature.*
- **Van Hasselt, H., Guez, A., Silver, D. (2016).** Deep Reinforcement Learning with Double Q-learning (DDQN). *AAAI.* arXiv:1509.06461.

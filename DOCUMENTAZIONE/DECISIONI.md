# Decisioni tecniche — Architecture Decision Records

Ogni decisione documenta: contesto → cosa abbiamo scelto → perché → conseguenze.  
Leggere prima di cambiare parametri o architettura.

---

## ADR-01: BETA_DECAY = 0.999 (non 0.995)

**Contesto:** In `paper_implementation`, `BETA_DECAY=0.995` portava ε a 0.05 già a episodio ~600 su 3000. L'agente smetteva di esplorare troppo presto, prima di aver visto abbastanza stati.

**Decisione:** `BETA_DECAY = 0.999`.

**Ragionamento:** Con 3000 episodi: `1.0 × 0.999^3000 ≈ 0.050`. Epsilon scende gradualmente per tutto il training, mantenendo esplorazione significativa fino alla fine.

| ep | ε (0.999) | ε (0.995) |
|----|-----------|-----------|
| 100 | 0.905 | 0.606 |
| 500 | 0.607 | 0.082 |
| 1000 | 0.368 | 0.007 |
| 3000 | 0.050 | ~0 |

**Conseguenza:** ε raggiunge il minimo a ep 3000, non a 600. Il training `feng_direct` ha confermato che a ep 1996 (ε=0.136) il primo episodio completo viene registrato — coerente con questa scelta.

---

## ADR-02: 16 spawn point su Maze 2 (non 8)

**Contesto:** La prima versione di `feng_direct` usava 8 spawn in zona centrale (clustered). Block 2 del training precedente mostrava 349/1400 crash a step=1 (24.9%) per imprecisione Gazebo nel teleport (±2-3cm).

**Decisione:** 16 spawn in 6 zone geografiche (A-F) coprendo tutta l'area di Maze 2 (x∈[-7,+6], y∈[-4,+6]).

**Ragionamento:**
1. **Copertura geografica:** l'agente vede le stesse sezioni di labirinto da angolazioni diverse → policy più robusta.
2. **Safety check:** dopo ogni teleport, `min_lidar >= 0.40m` obbligatorio (max 3 retry). Riduce crash step=1.
3. **16 vs 8:** Maze 2 ha ~15×13m con muri diagonali. 8 punti lasciavano zone F (inferiore) e E (superiore) mai visitate.

**Conseguenza:** Tutti 16 punti validati con `test_spawns.sh 2` (min LIDAR ≥ 0.43m). Crash step=1 nel training eliminati quasi completamente.

---

## ADR-03: No curriculum in `feng_direct`

**Contesto:** `paper_implementation` usava curriculum Maze 1 → Maze 2. Il curriculum falliva per: catastrophic forgetting, phase transition errata, epsilon troppo basso.

**Decisione:** training diretto su Maze 2 solo, nessun curriculum.

**Ragionamento:** L'obiettivo primario di `feng_direct` è replicare fedelmente Feng et al. (2021) nella sua forma più semplice, senza aggiunte non presenti nel paper. Il curriculum era una nostra aggiunta che ha complicato il debug. Prima si capisce se il metodo base funziona, poi si aggiungono elementi.

**Conseguenza:** 0% generalizzazione su Maze 1 e 3 — atteso per design. Per generalizzazione, il prossimo step è multi-maze training (con epsilon e threshold corretti). Vedere [NEXT_STEPS.md](NEXT_STEPS.md).

---

## ADR-04: Reward binario (+5 / -1000)

**Contesto:** `curriculum_learning` usava reward complessa (danger zone, steering penalty, space bonus). Mascherava mancanza di apprendimento reale.

**Decisione:** reward semplificata al minimo: +5 vivo, -1000 crash.

**Ragionamento:** Seguire il paper originale. Reward più semplice → più facile diagnosticare problemi. La reward complessa in `curriculum_learning` aveva prodotto buoni numeri in training ma policy non generalizante.

**Conseguenza nota (critica):** Il reward binario non dà segnale graduato di pericolo. L'agente non sa che avvicinarsi a 0.30m è più rischioso che stare a 2.00m. Contribuisce al 99.6% crash rate in training. **Reward shaping è il prossimo miglioramento prioritario.**

---

## ADR-05: Maze 3 = test-only, mai in training

**Contesto:** Serve un ambiente di valutazione che l'agente non abbia mai visto, per misurare generalizzazione vera.

**Decisione:** Maze 3 (`labirinto_10.world`) è escluso dal training. Usato solo in `start_test.sh`.

**Conseguenza:** Qualsiasi successo su Maze 3 è misura pulita di generalizzazione. Finora: 0% successo su tutti gli esperimenti — confermato overfitting strutturale del DRL.

---

## ADR-06: MSE loss (non Huber) ✅ CONFERMATA CORRETTA — 2026-05-10

**Contesto:** Il codice usa `nn.MSELoss()`. Mnih 2015 (DQN originale) usa Huber loss.

**Decisione:** mantenere MSE. Non sostituire con Huber.

**Ragionamento aggiornato (2026-05-10):** `fixed_feng` ha sostituito MSE con SmoothL1(δ=1) + clip=1.0. Risultato: avg100 < 0 dopo 3000 ep (vs `feng_direct` con MSE che raggiungeva +391). Feng 2021 usa MSE pura (Eq.5) — nessuna menzione di Huber. MSE è la scelta corretta per questo task e per questa scala di reward.

**Perché Huber(δ=1) peggiora:** con Q-values in [-1000, +5000] tutti i TD-error sono |e| >> δ=1 → loss sempre lineare → gradiente crash ≈ 1/batch → segnale ~10.000× più debole di MSE. Il δ=1 default di PyTorch è miscalibrato per questa scala.

**Se si vuole Huber in futuro:** δ deve essere proporzionale alla scala dei TD-error tipici (~100), non al default 1.0.

---

## ADR-07: GAMMA = 0.99 (non 0.999)

**Contesto:** Qualcuno ha proposto GAMMA=0.999 per orizzonte più lungo.

**Decisione:** mantenuto 0.99.

**Ragionamento:** Con GAMMA=0.99, l'orizzonte effettivo = `1/(1-0.99) = 100 step`. Con MAX_STEPS=1000, questo è ragionevole — non si bootstrappa su orizzonti impossibili. GAMMA=0.999 darebbe orizzonte 1000 step, uguale al limite episodio → Q-values tenderebbero a esplodere in early training.

---

## ADR-08: Gradient clip = 10.0 ✅ CONFERMATA CORRETTA — 2026-05-10

**Contesto:** Mnih 2015 (DQN originale) usa clip=1.0. Il codice usa 10.0.

**Decisione:** mantenere clip=10.0. Non abbassare a 1.0.

**Ragionamento aggiornato (2026-05-10):** `fixed_feng` ha testato clip=1.0. Risultato: avg100 < 0, peggiore di `feng_direct` con clip=10.0. Clip=1.0 è valido nel DQN originale perché Mnih usa reward clipping [-1,+1] — con reward clippata i gradienti sono piccoli e clip=1.0 non soffoca. Con reward +5/−1000 non clippato, clip=1.0 blocca quasi tutti gli aggiornamenti utili. Clip=10.0 permette aggiornamenti sostanziali senza esplosione dei gradienti, coerente con la scala del task.

**Feng 2021:** non menziona gradient clipping — probabilmente non lo usa affatto.

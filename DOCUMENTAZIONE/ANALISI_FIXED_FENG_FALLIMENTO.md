# Analisi fallimento `fixed_feng` — Diagnosi e letteratura

**Data:** 2026-05-10  
**Branch analizzato:** `fixed_feng` (3 commit sopra `feng_direct`)  
**Autore modifiche:** BoloM03 (Matteo Bolo)  
**Contesto:** training 3000 ep con modifiche hyperparametrali — avg100 rimasto negativo, loss bassa.

---

## 1. Le modifiche introdotte in `fixed_feng`

Commit `98e1b5b` ha applicato tre cambiamenti a `train_core.py`:

```python
# PRIMA (feng_direct — funzionante, avg100 = +391)
BATCH_SIZE = 64
self.loss_fn = nn.MSELoss()
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)

# DOPO (fixed_feng — fallito, avg100 < 0 dopo 3000 ep)
BATCH_SIZE = 256
self.loss_fn = nn.SmoothL1Loss()  # Huber, δ=1.0 default
torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
```

**Sintomi osservati:**
- avg100 ancora negativo a ep 3000 (vs `feng_direct`: avg100=+391)
- Loss bassa (apparente convergenza, ma policy non funzionante)
- Training percepito più lento

---

## 2. Cosa dice effettivamente il paper (Feng et al. 2021)

Lettura diretta del PDF conferma:

| Aspetto | Feng 2021 (paper) | fixed_feng (Matteo) |
|---|---|---|
| Loss function | MSE pura, Eq. 5: `L = 1/(2n) Σ(yi − Q)²` | SmoothL1 (Huber δ=1) |
| Gradient clipping | **Non menzionato** — non usato | 1.0 |
| Batch size | Non specificato (standard Keras default ~32-64) | 256 |
| PER | Testato, **scartato** — "lower value in the end" | Proposto come soluzione |

**Citazione esatta sul PER (§3.2, p.6):**
> *"The comparison between the DDQN with PER and the original DDQN based method indicated that the reward of DDQN with PER converged faster than the original DDQN but achieved a lower value in the end. Therefore, our obstacle avoidance method was developed based on DDQN."*

---

## 3. Diagnosi delle cause di fallimento

### CAUSA PRINCIPALE: Huber(δ=1) + grad_clip(1.0) ha azzerato il segnale di apprendimento

La combinazione è internamente incoerente per la scala di reward di questo task.

**Analisi gradiente per step:**

Con Q-values in range [-1000, +5000] e crash TD-error tipico ≈ 1000–1500:

- **SmoothL1 regime:** per `|error| > δ=1`, loss è **sempre lineare** → gradiente = `sign(error) × 1.0`. Quasi tutti gli errori (inclusi i crash) finiscono nel regime lineare.
- **Contributo crash per batch:** con BATCH=256 e ~1.4 crash/batch → gradiente crash ≈ `1.4/256 ≈ 0.005`
- **Con grad_clip=1.0:** il clipping non viene mai raggiunto (gradiente già 0.005 << 1.0)
- **Aggiornamento effettivo:** `LR × grad ≈ 0.00025 × 0.005 ≈ 0.000001` — praticamente zero

Con `feng_direct` (MSE + clip=10.0):
- Crash TD-error=1000 → gradiente MSE = `2×1000 = 2000` per campione
- Con media su batch 64: ~31 per step → clip a 10.0 attivo → aggiornamento efficace

**Conclusione:** il segnale di apprendimento dai crash (l'unica informazione realmente utile in questo task) era stato ridotto di ~10.000× rispetto a `feng_direct`.

**"Loss bassa" = conferma diagnostica, non di successo.** La rete converge rapidamente verso la predizione di Q-values costanti (negativi o prossimi a zero), raggiungendo equilibrio stabile con loss bassa ma policy inutile. Classic *bad fixed point*.

### ERRORE CONCETTUALE: grad_clip=1.0 richiede reward clipping

Nella DQN originale (Mnih et al. 2015), grad_clip=1.0 funziona perché **il reward viene clippato a [-1, +1]**. Con reward clipping, i TD-error non superano mai ~10-20, rendendo grad_clip=1.0 appropriato. Senza reward clipping (come nel nostro caso e in Feng), il clipping a 1.0 soffoca il segnale.

### PERCHÉ PER PEGGIORA IN QUESTO TASK

Con reward binario +5/−1000:
- TD-error crash: ~1000–1500
- TD-error survival: ~0–50

PER campiona proporzionalmente all'errore TD. Il ratio di campionamento crash/survival diventa ~200:1 invece di ~0.5:1 (uniform). La rete diventa **iper-conservativa**: impara "stai lontano da qualsiasi muro" ma perde la capacità di navigare i corridoi stretti dove ci deve necessariamente avvicinare alle pareti. Reward finale inferiore — esattamente l'osservazione di Feng.

### ERRORI NELL'ANALISI DI MATTEO (`ANALISI_PARAMETRI_FENG.md`)

1. **"Nel paper (Figura 3 e Sezione 3.3.2), gli autori specificano di aver ottenuto la vera stabilità usando il PER"** — FALSO. Il paper dice l'opposto: PER fu testato e scartato perché peggiora il reward finale.
2. **"MSE Loss genera uno shock ai pesi"** — vero come principio generale, ma Feng usa MSE e funziona. L'instabilità non era il problema reale di `feng_direct` (che raggiungeva avg100=+391).
3. **Il vero problema di `feng_direct` non era nei hyperparametri** ma nel gap prestazionale con il paper di Feng (vedi §4).

---

## 4. Il problema aperto: gap con i risultati di Feng

Anche `feng_direct` (fedele al paper, MSE, no Huber, no clip) mostra performance molto inferiori a Feng 2021:

| Metrica | `feng_direct` (noi) | Feng 2021 (paper) |
|---|---|---|
| Test su training maze | 10% successi (3/30 ep) | 0 collisioni in 5 min |
| Test su maze mai visti | 0% (0/30 ep) | Generalizza a 3 mappe reali diverse |
| Crash durante training | 99.6% | Non riportato |

Questo gap **non è spiegabile dai hyperparametri** (abbiamo la stessa configurazione). Le ipotesi per spiegarlo:

**Ipotesi A — Metriche non comparabili:** Feng valuta "collisioni in 5 minuti" (test continuo, robot può fermarsi o muoversi lentamente). Noi valutiamo "episodi da 500 step da spawn fissi". Se il robot di Feng impara a muoversi lentamente e con cautela, la sua metrica appare ottimale mentre la nostra (500 step = ~50 secondi a 10Hz) sembrerebbe fallimento.

**Ipotesi B — Difficoltà del maze:** Il Maze 2 nostro potrebbe avere passaggi più stretti o layout più complesso del Map 2 di Feng. Feng riporta 0 collisioni ma la traiettoria Figure 14 mostra percorso lungo le pareti del corridoio — suggerisce ambiente relativamente aperto.

**Ipotesi C — Dinamiche robot diverse:** STORM è skid-steer (trazione differenziale, frena bene). USV ha inerzia idrodinamica molto diversa: non frena, tende a slittare nelle curve. A parità di rete e reward, la policy apprende comportamenti subottimali per dinamiche diverse.

**Ipotesi D — Training insufficiente:** 3000 episodi potrebbero non essere sufficienti per il nostro ambiente più complesso. Feng riporta la curva reward come "slowly but stably increased" con β=0.999 — non riporta crash rate durante training né se il modello ha completamente convergito.

---

## 5. Approcci per la prossima iterazione

### Approccio A — Revert + analisi del gap [raccomandato come primo passo]
Tornare esattamente a `feng_direct` (MSE, no grad_clip, batch 64). Raccogliere più dati per spiegare il gap:
- Rieseguire test con metrica "5 minuti continuativi" (come Feng) invece di episodi discreti
- Misurare traiettorie del robot durante test

### Approccio B — Reward shaping potential-based
Aggiungere segnale di pericolo continuo senza cambiare l'algoritmo:
```python
# usv_logic.py
danger = max(0, 1.0 - min_lidar / DANGER_THRESHOLD)
reward = 5.0 - k * danger**2  # k da calibrare (es. 30)
# collision invariato: -1000
```
Feng identifica questo come "future work" (§6). Ng et al. (1999) garantisce invarianza della policy ottimale se lo shaping è potential-based.

### Approccio C — Multi-maze + più episodi
Feng §6: *"Further training and investigations will include a multi-stage training with a shared memory set and maps of gradually increased difficulty."* Alternare Maze 1 + Maze 2 ogni N episodi.

---

## 6. Riferimenti bibliografici consigliati

### Paper fondamentali (da leggere in questo ordine)

**[1] Feng et al. (precedente lavoro citato come [32] nel PDF)**  
Il paper di confronto DQN/DDQN/PER citato in §3.2. Contiene Figure 2-3 originali con la comparazione che spiega il rifiuto di PER. Non disponibile nel repository — da reperire.  
*Perché:* è la base empirica delle scelte algoritmiche di Feng 2021.

**[2] van Hasselt, Guez, Silver (2016) — "Deep Reinforcement Learning with Double Q-learning" (AAAI)**  
Paper DDQN originale. §4: hyperparameter analysis, gradient updates, relazione con DQN.  
*Perché:* spiega l'architettura del nostro agente e le scelte di design (perché due reti, perché θ⁻).

**[3] Mnih et al. (2015) — "Human-level control through deep reinforcement learning" (Nature)**  
DQN originale. §Methods: reward clipping [-1,+1] + grad_clip=1.0.  
*Perché:* spiega perché grad_clip=1.0 funziona in DQN classico ma richiede reward clipping — la condizione che fixed_feng non rispettava.

**[4] Schaul et al. (2016) — "Prioritized Experience Replay" (ICLR)**  
PER con importance sampling weights (IS correction).  
*Perché:* §4.3 spiega perché senza IS correction PER distorce la distribuzione target e può peggiorare le prestazioni con reward sparso e molto sbilanciato. Risponde direttamente a "perché PER peggiora in Feng".

**[5] Henderson et al. (2018) — "Deep Reinforcement Learning That Matters" (AAAI)**  
Studio su riproducibilità in RL.  
*Perché:* §3-4 dimostrano che la stessa implementazione con stessi hyperparametri può dare risultati radicalmente diversi a causa di random seed, environment differences, implementation details. Risponde al gap feng/nostro.

**[6] Ng, Russell et al. (1999) — "Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping" (ICML)**  
Teorema fondamentale del reward shaping.  
*Perché:* dimostra formalmente quando aggiungere un termine di reward non cambia la policy ottimale (shaping potential-based). Base teorica per Approccio B.

**[7] Tai, Paolo, Liu (2017) — "Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation" (IROS)**  
RL per robot navigation in ambienti simili al nostro.  
*Perché:* usa reward shaping continuo (distanza dal target + pericolo) invece di reward binario. Confronto diretto con il nostro approccio.

### Paper aggiuntivi (se si vogliono approfondire)

**[8] Goyal et al. (2017) — "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" (arXiv)**  
Linear scaling rule: LR deve scalare con batch size.  
*Perché:* spiega H2 (LR non aggiustato per BATCH=256) con fondamento matematico.

**[9] Hessel et al. (2018) — "Rainbow: Combining Improvements in Deep Reinforcement Learning" (AAAI)**  
Ablation study sistematico di 6 estensioni DQN.  
*Perché:* Figura 6 mostra quale singolo miglioramento conta di più. PER è importante ma solo in combinazione con IS correction.

---

## 7. Sintesi — cosa fare e cosa non fare

**NON fare:**
- Aggiungere Huber loss senza aumentare δ proporzionalmente alla scala dei Q-values (δ=1 è sbagliato per Q in [-1000, +5000])
- Applicare grad_clip < 5.0 senza reward clipping
- Implementare PER — Feng lo ha testato ed è peggio per questo task specifico

**FARE:**
- Revert a `feng_direct` baseline prima di qualsiasi altra modifica
- Capire il gap con Feng analizzando le traiettorie e la metrica di test
- Se si aggiunge reward shaping, usare forma potential-based (Ng 1999)
- Leggere il paper [32] di Feng (lavoro precedente) per capire la comparazione DQN/DDQN/PER

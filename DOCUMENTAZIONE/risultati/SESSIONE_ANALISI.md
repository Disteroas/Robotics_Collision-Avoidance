# Sessione di Analisi — corrections_claude branch

**Data:** 2026-05-06  
**Branch:** `corrections_claude`  
**Obiettivo:** Capire perché il training non ha prodotto risultati attesi e identificare le cause root.

---

## 1. Punto di partenza

Il training su `corrections_claude` era stato completato (3000 episodi, curriculum Phase1→Phase2). I risultati non erano quelli attesi. La sessione di analisi è partita dal leggere i CSV prodotti dal training.

**File analizzati:**
- `src/my_usv/scripts/training_log.csv` — 3000 righe, log episodio per episodio
- `src/my_usv/scripts/test_results.csv` — 90 righe (30 ep × 3 maze), best_ddqn_model.pth a ε=0

---

## 2. Analisi quantitativa dei CSV

### 2.1 Risultati test

| Maze | Successi/30 | Steps medi | avg_lidar |
|------|------------|------------|-----------|
| Maze 1 (9a) | **30/30 (100%)** | 500 (max) | 2.95 m |
| Maze 2 (9b) | **0/30 (0%)** | 77 | 0.83 m |
| Maze 3 (10) | **0/30 (0%)** | 67 | ~1.07 m |

**Osservazione critica:** il robot naviga perfettamente maze 1, ma crasha immediatamente su maze 2 e maze 3. La differenza di `avg_lidar` (2.95m vs 0.83m) era apparentemente indicativa di corridoi stretti in maze 2.

### 2.2 Statistiche training

| Maze | Episodi | Successi | Crash | Success rate |
|------|---------|----------|-------|-------------|
| Maze 1 | 1100 | 601 | 499 | **54.6%** |
| Maze 2 | 1900 | 80 | 1820 | **4.2%** |

- **Phase 2 triggered:** intorno a ep ~380 (avg50_maze1 > 1500)
- **Phase 2 iniziata:** ep 501 (primo blocco maze 2 dopo il trigger)
- **Epsilon a inizio Phase 2:** ε ≈ 0.08
- **avg100 finale (maze 2):** −618

**Timeline del learning:**
- Ep 1–274: tutti crash su maze 1 (pura esplorazione)
- Ep 275: **primo successo** (1000 step, reward ≈ +6500)
- Ep 329–382: prime sequenze consecutive di successi → trigger Phase 2
- Ep 501–3000: Phase 2 (70% maze 2, 30% maze 1). Maze 2: 4.2% successi.

---

## 3. Analisi geometrica dei maze

Per capire se "avg_lidar basso = corridoio stretto", abbiamo letto i file SDF world e generato mappe ASCII dei labirinti.

**Script usato:**
```python
# Parsing XML dei file .world in src/my_usv/worlds/
import math, xml.etree.ElementTree as ET
# estrae posizioni/dimensioni muri → render ASCII auto-scaled
```

**Risultati:**

### Maze 1 (labirinto_9a.world)
- Area: ~8×8m
- Muri: 20 totali, prevalentemente **axis-aligned** (rettilinei)
- Spawn: (x=−3, y=−5, yaw=90°) — fuori dal maze, il robot entra dal basso
- Corridoi: larghi, aperti, struttura rettangolare

### Maze 2 (labirinto_9b.world)
- Area: ~15×13m (molto più grande di maze 1)
- Muri: 31 totali, **25/31 (81%) DIAGONALI**
- Spawn: (x=−6, y=0, yaw=0°) — dentro il maze, circondato da muri diagonali
- Corridoi: geometricamente non stretti, ma con pareti diagonali

### Maze 3 (labirinto_10.world) — test only
- Area: ~10×8m
- Muri: 13 totali, con diagonale principale
- Spawn: (x=−2, y=−1, yaw=0°)

**Conclusione:** maze 2 non è "stretto" nel senso fisico. Il basso `avg_lidar=0.83m` durante il test riflette il fatto che il robot crasha in 77 step senza mai navigare — misura la posizione di crash, non la larghezza media dei corridoi.

**Il vero problema:** maze 2 ha l'81% dei muri diagonali. Durante Phase 1 (500 ep), il robot ha visto SOLO muri axis-aligned di maze 1. I pattern LIDAR di muri diagonali sono completamente diversi → **distributional shift** totale.

---

## 4. Diagnosi: perché il robot va dritto nel muro

In maze 3, il robot crasha al primo muro che ha davanti ("condizione di mare aperto"): **naviga perfettamente lo spazio libero, poi ignora il muro**.

Questo rivela che il robot ha imparato:
- "In maze 1, dalla posizione X con yaw Y, fai azione Z" → **memorizzazione path-specific**
- NON ha imparato: "quando LIDAR mostra muro davanti, gira"

**Causa:** spawn fisso per tutti i 3000 episodi. Il robot non può fare altro che memorizzare un percorso specifico.

**Perché il segnale di pericolo non ha aiutato:**
Con `FRONT_DANGER=3.0m`, a 2.5m dal muro la penalità è `20×(0.5/3.0)²=0.56`. Il `space_bonus` alla stessa distanza vale `+0.8`. Reward netto ancora positivo → il robot non riceve segnale chiaro per girare finché non è troppo vicino. Poi crasha.

---

## 5. Riletura del paper originale

**Paper:** *"A Collision Avoidance Method Based on Deep Reinforcement Learning"* — Feng, Sebastian, Ben-Tzvi, Robotics 2021.

### Differenze critiche trovate

| Feature | Paper | Nostra implementazione |
|---------|-------|------------------------|
| Reward | `+5 / −1000` (solo) | `5 + space_bonus − steering − danger` |
| Spawn | **RANDOM ogni episodio** | **Fisso** (x=−3, y=−5, yaw=1.57) |
| ε decay β | **0.999** (best result) | **0.995** (worst result nel paper) |
| Training | 1 maze complesso | 2 maze + curriculum |
| Generalization | Real world + unseen maps ✓ | Solo maze 1 ✓ |

**Citazione chiave (Algorithm 1, linea 3):**
> *"Put the visual robot at a **random position** in the 3D world"*

Lo spawn random è il meccanismo principale che previene l'overfitting e produce generalizzazione.

**Epsilon:** il paper testa β=0.999/0.998/0.997. Risultato migliore con β=0.999 (0 collisioni in test). Noi usiamo β=0.995, il loro caso peggiore (2 collisioni).

Con β=0.999, ε raggiunge 0.05 a ~3000 ep → esplorazione massima per tutto il training.  
Con β=0.995, ε raggiunge 0.05 a ~600 ep → 2400 ep quasi greedy su Q-network mal calibrato.

### Le nostre "migliorie" che hanno peggiorato

1. **Space bonus** → robot incentivato a stare in spazio aperto. In maze 2 corridoi, reward sempre basso → segnale confuso.
2. **Front danger 3.0m** → scatta quasi sempre in maze 2 → robot non sa distinguere "navigo bene" da "navigo male".
3. **Steering penalty** → penalizza le svolte → comportamento opposto a quello necessario in maze diagonale.
4. **Spawn fisso** → memorizzazione invece di generalizzazione.
5. **β=0.995** → esplorazione insufficiente.

---

## 6. Conclusioni

### Root causes identificate (ordine di impatto)

1. **Spawn fisso** — causa principale del mancato apprendimento generalizzato. Il robot ha memorizzato maze 1, non ha imparato a evitare ostacoli.

2. **β=0.995 troppo aggressivo** — esplorazione finisce a ~600 ep. Phase 2 inizia a ep 501 con ε≈0.08: troppo poco per esplorare maze 2 mai visto.

3. **Reward function troppo complessa** — space bonus, front danger e steering penalty sono componenti tutte calibrate per maze 1 aperto. In maze 2, producono segnali fuorvianti.

4. **Curriculum su base rotta** — Phase 2 con 1900 ep di maze 2 a ε basso, policy già specializzata per maze 1 → nessun apprendimento reale.

### Cosa funziona

- L'architettura DDQN (50→300→300→11) è corretta (identica al paper).
- LIDAR processing (512→50 bin, min-pooling) è corretta.
- Maze 1 è stato risolto con successo (100% in test).

---

## 7. Piano di azione

Tornare all'approccio del paper + estensioni:

| Parametro | Attuale | Proposto |
|-----------|---------|----------|
| Reward | Complessa (5 componenti) | `+5 / −1000` |
| Spawn | Fisso | **Random** (lista precompilata) |
| β (ε decay) | 0.995 | **0.999** |
| Training maze | Maze 1 (Phase1) + Maze 2 (Phase2) curriculum | **Maze 1 + 2, random 50/50, no curriculum** |
| Episodi | 3000 | 3000 |

**Spawn random:** lista precompilata di posizioni valide (non dentro muri) per maze 1 e maze 2. Il robot sceglie casualmente ogni episodio tra tutte le posizioni disponibili in tutti e due i maze.

**Maze 3 rimane test-only.**

---

## 8. File prodotti da questa sessione

| File | Descrizione |
|------|-------------|
| `risultati/analisi.py` | Script Python per analisi e plot automatica di training_log.csv e test_results.csv |
| `risultati/SESSIONE_ANALISI.md` | Questo documento |
| `risultati/plots/training_reward.png` | Reward grezzo + avg100 + success rate per maze |
| `risultati/plots/epsilon_loss.png` | Epsilon decay e loss nel tempo |
| `risultati/plots/steps_per_episode.png` | Steps per episodio per maze |
| `risultati/plots/phase_transition_zoom.png` | Zoom sulla transizione Phase1→Phase2 |
| `risultati/plots/maze2_detail.png` | Distribuzione reward e steps maze 2 |
| `risultati/plots/test_results.png` | Confronto test su maze 1/2/3 |

Per rigenerare tutti i plot:
```bash
python risultati/analisi.py
```

---

## 9. Riferimenti

- Paper: Feng S., Sebastian B., Ben-Tzvi P. — *"A Collision Avoidance Method Based on Deep Reinforcement Learning"* — Robotics 2021, 10, 73.
- Branch analizzato: `corrections_claude` (commit head: `cae60c8`)
- CSV training: `src/my_usv/scripts/training_log.csv` (3000 episodi, 2026-05-06)
- CSV test: `src/my_usv/scripts/test_results.csv` (90 episodi, 2026-05-06)

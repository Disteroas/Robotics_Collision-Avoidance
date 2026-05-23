# Guida operativa — Simulazioni rigorose (seed, 4 PC, reporting)

**A chi serve:** tutti i membri del gruppo che lanciano training o test.
**Perché esiste:** abbiamo scoperto che due run con codice identico davano risultati molto diversi (M1 0%↔50%, M3 0%↔23%) perché **non c'era controllo del seed**. Da qui in poi ogni esperimento deve essere riproducibile e confrontabile. Questa guida dice **esattamente cosa fare**, passo per passo.

> Riferimenti interni: `ANALISI_STRATEGICA_2026-05-21.md` (perché siamo arrivati a questo), `docs/superpowers/specs/2026-05-22-rigorous-simulation-base-design.md` (design), `CLAUDE.md` (riferimento rapido).

---

## 0. TL;DR (la versione da 30 secondi)

1. Ogni run ha un **seed** (`--seed=N`) e una **config** (`--config=nome`). Gli artefatti finiscono in `runs/<config>/seed_<N>/`.
2. Un singolo seed **non vale come risultato**. Servono **almeno 3-5 seed** della stessa config.
3. I 4 PC servono a girare i seed **in parallelo**.
4. Alla fine si aggregano con `aggregate_seeds.py` e si riporta **media ± deviazione standard (+ IQM, + intervallo di confidenza)**, **mai il valore massimo**.
5. Prima di un `--reset`, gli artefatti vengono **salvati automaticamente** in `ANALISI_TRAINING/`. Non disattivare questa protezione.

---

## 1. Concetti chiave

### Seed
Il seed fissa tutte le sorgenti casuali controllabili (pesi rete, campionamento replay, ε-greedy, scelta spawn, rumore LIDAR). Si imposta con `--seed=N`.

⚠️ **Importante:** Gazebo (fisica + timing ROS) **resta non deterministico**. Quindi il seed **non** dà run identiche bit-a-bit. Quello che dà è: la varianza diventa **misurabile e attribuibile** invece che invisibile. Due seed diversi → due run legittimamente diverse di cui possiamo misurare la dispersione.

### Config
Etichetta dell'esperimento (`--config=nome`). Tutti i seed di uno stesso esperimento condividono la config. Esempi di nomi sensati:
- `r_alpha` — reward R-α attuale (baseline DDQN onesto)
- `r_directional` — futura reward direzionale
- `td3` — futura traccia TD3

Una config = **una variabile cambiata**. Se cambi reward E rete insieme, non saprai a cosa attribuire il risultato.

### Layout artefatti
```
runs/<config>/seed_<N>/
  checkpoint.pkl        # stato training (pesi, replay, epsilon, seed)
  best_model.pth        # miglior modello (usato dal test)
  training_log.csv      # 1 riga per episodio + colonna crash_sector
  eval_summary.csv      # 1 riga per maze: success_rate, reward, steps
  eval_steps_m{1,2,3}.csv    # 1 riga per STEP: azione, Q, distanze settore
  eval_crashes_m{1,2,3}.csv  # 1 riga per crash: settore, distanza, ultime 5 azioni
  run_meta.json         # seed, git_sha, hostname, timestamp, criterio successo
```
`runs/` è in `.gitignore` (sono GB di artefatti). Si versiona **solo** l'aggregato in `ANALISI_TRAINING/`.

---

## 2. Prerequisiti (una volta per PC)

1. **Docker Desktop** avviato (backend WSL2).
2. **VcXsrv/XLaunch** attivo con "Disable access control" (serve solo se vuoi la GUI Gazebo; il training/test gira headless).
3. Immagine + build ROS già fatti almeno una volta:
   ```bash
   docker build -t usv_rl_project .
   docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash
   # dentro: colcon build && exit
   ```
4. Tutti i comandi si lanciano da **Git Bash**, dalla **root del progetto**.

Verifica veloce che Docker risponda:
```bash
docker ps
```

---

## 3. Training (un seed)

```bash
./start_train_multimaze.sh --seed=0 --config=r_alpha
```

- Gira 5000 episodi (M1+M2, ratio 1:2), 25 blocchi da 200. Gazebo si riavvia a ogni blocco.
- Tempo stimato: **~8 ore**.
- Output: `runs/r_alpha/seed_0/` (`checkpoint.pkl`, `best_model.pth`, `training_log.csv`).
- Log per blocco: `logs/multimaze_block_*.log`.

Riprende automaticamente da `checkpoint.pkl` se lo rilanci con gli stessi `--seed`/`--config` (utile dopo un'interruzione).

### Reset (ripartire da zero per quella config/seed)
```bash
./start_train_multimaze.sh --seed=0 --config=r_alpha --reset
```
Prima di cancellare, **fa un backup automatico** di `runs/r_alpha/seed_0/` in `ANALISI_TRAINING/<data>/pre_reset_r_alpha_seed_0/`. Se il backup fallisce, il reset **si annulla** (per non perdere dati). **Non aggirare questa protezione** — abbiamo già perso i CSV grezzi del Round 1 così.

---

## 4. Test / valutazione (un seed)

```bash
./start_test.sh --seed=0 --config=r_alpha --reps=30
```

- Valuta `runs/r_alpha/seed_0/best_model.pth` su tutti e 3 i maze, ε=0.0 (greedy puro).
- **Round-robin**: ogni spawn viene testato esattamente `--reps` volte, in sequenza fissa. Quindi:
  - M1 = 2 spawn × 30 = **60 episodi**
  - M2 = 6 spawn × 30 = **180 episodi**
  - M3 = 1 spawn × 30 = **30 episodi**
- Coverage bilanciata e confrontabile tra seed e tra config.

### Cosa produce e cosa guardare
| File | A cosa serve |
|---|---|
| `eval_summary.csv` | Il numero che conta: success_rate per maze. È l'input dell'aggregazione. |
| `eval_steps_m<N>.csv` | Diagnostica fine: per ogni step l'azione scelta, i Q-value (`q_chosen`, `q_max`, `q_spread`), le distanze `front/left/right/min_lidar`. Serve a capire **cosa fa** la policy. |
| `eval_crashes_m<N>.csv` | Per ogni crash: `crash_sector` (front/left/right), distanza, **ultime 5 azioni**. Serve a capire **perché** si schianta. |
| `run_meta.json` | Provenienza: seed, commit git, hostname, timestamp. Sempre allegato ai risultati. |

> Opzione `--log-q-full`: aggiunge tutti gli 11 Q-value per step (CSV ~2× più grandi). Usala solo quando devi analizzare a fondo la policy.

---

## 5. Multi-seed e i 4 PC (il punto centrale)

### Quanti seed
**Minimo 3, idealmente 5.** Henderson 2018 e Agarwal 2021 mostrano che meno di così non distingue segnale da rumore.

### Come dividerli sui 4 PC
Ogni seed è un training indipendente (~8h). Per una config con 5 seed, parallelizza così:

| PC | Seed assegnati | Comando |
|---|---|---|
| PC1 | 0, 4 | `./start_train_multimaze.sh --seed=0 --config=r_alpha` poi `--seed=4` |
| PC2 | 1 | `./start_train_multimaze.sh --seed=1 --config=r_alpha` |
| PC3 | 2 | `./start_train_multimaze.sh --seed=2 --config=r_alpha` |
| PC4 | 3 | `./start_train_multimaze.sh --seed=3 --config=r_alpha` |

Così 4 seed finiscono in ~8h e il quinto in ~16h sul PC1.

### Hardware come confound (leggere)
Idealmente i seed di una config girano sullo **stesso PC**, perché PC diversi aggiungono una piccola varianza hardware sopra quella da seed. Nella pratica parallelizziamo sui 4 PC per velocità: va bene **a patto che** `run_meta.json` registri l'hostname (lo fa in automatico). La varianza da seed domina su quella hardware, quindi il confronto resta valido. Solo se due config risultano **molto vicine** (dentro le bande di incertezza) conviene ri-girare i seed decisivi sullo stesso PC per pulizia.

### Raccogliere i risultati dai 4 PC
Ogni PC produce la sua `runs/<config>/seed_<N>/`. Per aggregare, copia le cartelle `seed_*` di tutti i PC nella stessa `runs/<config>/` su una sola macchina (chiavetta, share di rete, o git LFS se configurato — **non** git normale, `runs/` è ignorato). Bastano i file `eval_summary.csv` per l'aggregato; copia anche `eval_steps`/`eval_crashes` se vuoi la diagnostica.

---

## 6. Aggregazione e reporting

Dopo aver eseguito il **test** (`./start_test.sh`) per ogni seed:

```bash
python3 src/my_usv/scripts/aggregate_seeds.py \
  --config r_alpha \
  --output ANALISI_TRAINING/$(date +%Y_%m_%d)/aggregate_r_alpha.csv
```

Legge tutti i `runs/r_alpha/seed_*/eval_summary.csv` e produce, per maze:
- **media ± deviazione standard** del success_rate sui seed,
- **IQM** (Inter-Quartile Mean, robusto agli outlier),
- **intervallo di confidenza 95%** via bootstrap.

### Regola d'oro del reporting
- Riporta sempre `media ± std (IQM; 95% CI) su N seed`.
- **MAI** riportare il massimo tra i seed come se fosse "il risultato".
- Con **un solo seed** lo std è `NaN` (varianza indefinita): è un segnale che ti manca il multi-seed, non un risultato pubblicabile.

Esempio di frase corretta in un report:
> "M2: 71.2% ± 9.4 (IQM 73.0%; 95% CI [62, 80]) su 5 seed."

Esempio **sbagliato**:
> "M2: 83% (best run)." ← rumore spacciato per segnale.

---

## 7. Diagnostica con i log per-step

Il logging serve a rispondere a "**perché** si schianta?", non solo "**quanto** spesso".

Esempio reale dallo smoke test: in M1 spawn P2 `(1.0,-1.0)`, `eval_crashes_m1.csv` mostrava
```
crash_sector=right, last_actions="10,10,10,10,10"
```
→ la policy sceglie ostinatamente l'azione 10 (sterzata massima a destra) e si schianta a destra. Senza il logging avremmo visto solo "0% su P2"; con il logging sappiamo che è un problema di **scelta dell'azione**, non di percezione cieca.

Domande tipiche e dove guardare:
- "L'agente va dritto contro il muro?" → `eval_steps`: guarda `action` (5 = dritto) vs `front_dist` che cala.
- "La policy è confusa (Q tutti simili)?" → `q_spread` piccolo = poca differenza tra azioni.
- "Su quale settore muore di più?" → conta i `crash_sector` in `eval_crashes`.

---

## 8. Regole d'oro / cosa NON fare

✅ **Fai:**
- Una variabile per config (isola la causa).
- ≥3-5 seed prima di trarre conclusioni.
- Stesso protocollo di eval (`--reps` uguale) tra config che confronti.
- Allega sempre `run_meta.json` ai risultati.
- Backup CSV prima di ogni reset (è automatico — non disattivarlo).

❌ **Non fare:**
- Confrontare due config con `--reps` o set di maze diversi (confronto "confounded": non puoi attribuire il delta).
- Riportare un singolo seed o il massimo.
- Cambiare reward + rete + action-space nello stesso esperimento.
- Cancellare `runs/` a mano senza backup.
- Girare altri seed di una config che sai già rotta (es. cinematica F1) — sprechi PC-ore.

---

## 9. Checklist operativa (copia-incolla)

**Nuovo esperimento `r_alpha`, 5 seed su 4 PC:**

```bash
# --- su ciascun PC, training (seed assegnato dalla tabella §5) ---
./start_train_multimaze.sh --seed=<N> --config=r_alpha
# ... attendere ~8h ...

# --- su ciascun PC, test ---
./start_test.sh --seed=<N> --config=r_alpha --reps=30

# --- raccogli tutte le runs/r_alpha/seed_*/ su una macchina, poi: ---
python3 src/my_usv/scripts/aggregate_seeds.py \
  --config r_alpha --output ANALISI_TRAINING/$(date +%Y_%m_%d)/aggregate_r_alpha.csv

# --- versiona SOLO l'aggregato ---
git add ANALISI_TRAINING/$(date +%Y_%m_%d)/aggregate_r_alpha.csv
git commit -m "data(r_alpha): aggregato 5 seed"
```

---

## 10. Troubleshooting

| Sintomo | Causa probabile / fix |
|---|---|
| `❌ Modello non trovato` al test | Training non completato o `--config`/`--seed` sbagliati: il test cerca `runs/<config>/seed_<N>/best_model.pth`. |
| Gazebo "crashato" / non parte | VcXsrv non attivo o Docker non avviato; aumenta `GAZEBO_WAIT` nello script; rilancia (`docker rm -f usv_container` se resta appeso). |
| Test parte ma 0% ovunque | Controlla che il modello sia davvero quello giusto (`run_meta.json` → `git_sha`); guarda `eval_crashes` per capire il settore. |
| Reset "abortito" | Il backup automatico è fallito (disco pieno/permessi). Libera spazio: è una protezione, non un bug. |
| Risultati di due seed identici sospetti | Verifica che `--seed` sia effettivamente diverso (lo trovi in `run_meta.json`). |
| `std = nan` nell'aggregato | Hai un solo seed per quella config. Aggiungine altri. |

---

*Documento operativo. Tienilo aperto mentre lanci gli esperimenti. Se cambi lo script o aggiungi una traccia (es. TD3), aggiorna questa guida.*

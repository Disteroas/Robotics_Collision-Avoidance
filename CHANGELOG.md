# Changelog

Organizzato per fase di sviluppo, dal più recente. Ogni voce riporta cosa è cambiato e perché.

---

## merge11_05 — 2026-05-11

Branch: multi-maze interleaved training. Spawn M1 validati. Training pronto.

| Commit | Descrizione |
|--------|-------------|
| `f03f56e` | docs(analysis): P2 validato — entrambi i punti M1 confirmed |
| `568435e` | docs(analysis): plot M1 spawn aggiornato a 2 punti P1/P2 |
| `8a935a1` | fix(spawn): M1 ridotto a 2 punti — P1 (-2.9,-2.0,N), P2 (1.0,-1.0,N) heading N |
| `ceaf5bd` | docs: aggiornamento ESPERIMENTI, INDEX, NEXT_STEPS, CHANGELOG post-analisi M1 |
| `a284c0f` | docs: commento GAZEBO_SPEED=5 (confermato stabile in randomSpawn) |
| `0487944` | fix: rimozione `--rm` (race con docker exec), aggiunto trap INT/TERM, pulizia reset block |
| `7c03890` | feat: `start_train_multimaze.sh` — 25 blocchi × 200 ep, pattern M1/M2/M2, --reset flag |
| `eba4d25` | docs: docstring `train.py` aggiornato per multimaze + --total-ep |
| `e34ac1a` | refactor: rimossa phase transition logic da `train.py`, aggiunto `--total-ep` CLI arg |
| `e37ddba` | style: allineamento SPAWN_LISTS[1], annotazione Zone E (near boundary) |
| `892ab98` | feat: `usv_env.py` SPAWN_LISTS[1] espanso da 8 a 16 punti → ridotto a 2 post-analisi |
| `2fda3f5` | docs: spec e piano implementazione multi-maze training |

**Design:** 5000 ep, BETA_DECAY=0.999, reward complessa invariata, no curriculum.  
**Spawn M1:** 2 punti (canali lat. 1.50m < r_min 1.56m: U-turn impossibile; zona sud y<-3.81m senza muri).  
**Ipotesi:** diversità maze (Cobbe 2019) + random spawn (Tobin 2017) = generalizzazione M3.  
**Spec:** `docs/superpowers/specs/2026-05-11-multimaze-training-design.md`

---

## fixed_feng — 2026-05-09/10

Branch da BoloM03 (Matteo). Tentativo di stabilizzare training con fix hyperparametrali. Fallito.

| Commit | Descrizione |
|--------|-------------|
| `635112d` | `DOCUMENTAZIONE/ANALISI_PARAMETRI_FENG.md`: analisi batch/loss/clip (contiene errori — vedi sotto) |
| `3575fcf` | Aggiornamento `ANALISI_PARAMETRI_FENG.md`: aggiunta sezione Huber vs MSE |
| `98e1b5b` | `train_core.py`: BATCH_SIZE 64→256, MSELoss→SmoothL1, grad_clip 10→1.0 |
| `d68c115` | Aggiunto `labirinto_custom.world` + `muri_mix/` (non wired in train/test) |

**Training completato:** 3000 ep. Avg-100 rimasto negativo. Loss bassa (bad fixed point).  
**Causa:** Huber(δ=1)+clip=1.0 riduce segnale di apprendimento dei crash di ~10.000× vs feng_direct. Feng 2021 usa MSE pura, nessun grad_clip, nessun PER (PER testato da Feng: reward finale peggiore).  
**Errori in ANALISI_PARAMETRI_FENG.md:** PER indicato come soluzione ma Feng lo ha scartato; MSE indicata come instabile ma funziona nel paper.  
**Analisi completa:** `DOCUMENTAZIONE/ANALISI_FIXED_FENG_FALLIMENTO.md`

---

## feng_direct — 2026-05-08/10

Branch base. Implementazione diretta di Feng et al. (2021): training su Maze 2, no curriculum.

| Commit | Descrizione |
|--------|-------------|
| `a639323` | Analisi fallimento `fixed_feng`: diagnosi matematica + 9 reference bibliografiche |
| `d34ed18` | Fix report: rimossa CAUSA 1 errata (goal) — Feng 2021 usa stato LIDAR-only |
| `2d56b74` | Riorganizzazione repo: `analysis/`, `results/`, `models/`, `CHANGELOG` |
| `08b8049` | Aggiunta suite documentazione (`DOCUMENTAZIONE/INDEX, ARCHITETTURA, ESPERIMENTI, DECISIONI, NEXT_STEPS, TROUBLESHOOTING`) |
| `4bbc476` | Riorganizzazione .md in `DOCUMENTAZIONE/`. Fix bug `test.py`: `reset_environment()` passava sempre `maze_id=1` → spawn errato su Maze 2/3 |
| `998f8b0` | Fix: log LIDAR INFO stampato una volta sola all'avvio, non per ogni episodio |
| `b7e0a2c` | Spawn C2 → (-7.0, 5.0) e F3 → (6.0, 6.0) per copertura geografica migliore |
| `36698fb` | Aggiustamento 5 coordinate dopo validazione Gazebo (16/16 OK, min LIDAR ≥ 0.43m) |
| `6121ffa` | Fix critico: `accepting_scans=True` garantito dopo retry loop esaurito |
| `69cc8ca` | Safety check post-teleport: retry se `min_lidar < 0.40m` (max 3 tentativi) |
| `a2c8003` | Fix test: stub ROS2 assegnati per modulo, non prodotto cartesiano |
| `2ff6530` | Spawn Maze 2 espansi: 8 punti clustered → 16 punti in 6 zone (A-F) |
| `6d79d7d` | GUIDA_OPERATIVA riscritta per feng_direct |
| `bb2ef44` | Fix: plugin `gazebo_ros_state` in `labirinto_10.world`, trap EXIT in GUI script |
| `5a55a25` | Aggiunto `start_train_direct.sh`, `start_test_gui.sh` |

**Training completato:** 3000 ep, Maze 2. Avg-100 finale: +391. Test: 10% successi Maze 2, 0% Maze 1/3.  
**Analisi:** `DOCUMENTAZIONE/report_feng_direct.md` | **Plot:** `analysis/plots/feng_direct/`

---

## paper_implementation — 2026-05-07

Tentativo curriculum learning (Maze 1 Phase 1 → Maze 2 Phase 2). Fallito.

| Commit | Descrizione |
|--------|-------------|
| `5c1dd45` | Reset epsilon a 0.5 su Phase 2. Pass `maze_id` a `env.reset()` |
| `f3278eb` | Spawn per-episodio via `SetEntityState` teleport |
| `8aa6b61` | `BETA_DECAY` 0.995 → 0.999 per curva di decadimento su 3000 ep |
| `53e513a` | Reward semplificata: +5/−1000 (Feng 2021) |
| `5c1dd45` | Init da `curriculum_learning` |

**Training completato:** 6115 ep totali. Risultati test: crash >85% su tutti i maze.  
**5 cause fallimento:** ε troppo basso a ep 600, phase transition su avg reward, catastrophic forgetting, spawn fisso, reward densa senza apprendimento.  
**Analisi:** `DOCUMENTAZIONE/risultati/PAPER_IMPLEMENTATION_SESSION.md`

---

## corrections_claude / gym_env — 2026-05-05/06

Refactor codice e aggiunta wrapper Gymnasium.

| Commit | Descrizione |
|--------|-------------|
| `f6c00d1` | `train_gym.py`: training DDQN via Gymnasium |
| `73815a5` | `UsvGymEnv`: wrapper Gymnasium con action space discreto e continuo |
| `a4d456b` | Test suite per `UsvGymEnv` (10 test, TDD) |
| `b435475` / `d293d2e` | Spec e piano implementazione wrapper Gymnasium |

---

## curriculum_learning — 2026-05-04/05

Prima versione con curriculum Phase 1 → Phase 2 e reward complessa.

| Commit | Descrizione |
|--------|-------------|
| `37e6c55` | Script analisi training e report sessione |
| `cae60c8` | `TRAINING_GUIDE.md` per `corrections_claude` |

**Risultati:** crash rate ~73% su Maze 2 dopo 3000 ep. Reward complessa (danger zone, steering penalty) mascherava mancanza apprendimento reale.

---

## fast_sim / main — 2026-04-30 / 2026-05-03

Setup infrastruttura base: Docker, ROS 2, Gazebo accelerato, script bash.

| Commit | Descrizione |
|--------|-------------|
| `928cf9d` | Branch `fast_sim`: Gazebo 4×, `patch_world.py`, script headless |
| `31cc7ae` | Script bash per simulazione senza GUI |
| `55bb815` | Tutorial DDQN |
| `51b2a81` | Prima versione Docker + RL completa e pulita |
| `71eb489` | Initial commit |

# Repository Status — UGV DDQN Collision Avoidance

**Data:** 2026-05-06  
**Repo:** [Disteroas/Robotics_Collision-Avoidance](https://github.com/Disteroas/Robotics_Collision-Avoidance)

---

## Branch attivi

| Branch | Descrizione | Stato |
|---|---|---|
| `main` | Baseline originale. DDQN + reward semplice (+5/−1000 + danger zone 1.5m). Spawn fisso. β=0.999. | Stabile |
| `curriculum_learning` | Tentativo curriculum progressivo Phase1→Phase2 (Maze 1 → Maze 2). Reward complessa (space_bonus, front/side danger, steering penalty). 3000 ep completati. | **Analizzato** — risultati negativi documentati |
| `fast_sim` | Simulazione accelerata (Gazebo 4×). Reward più vicina al paper: ramp lineare [0.25m, 1.0m] → +5/−1000. β=0.999. Infrastruttura: patch_world.py, start_sim.sh, start_train.sh. | Pronto come base per prossimo training |
| `gym_env` | Tutto di `fast_sim` + wrapper Gymnasium. Aggiunge usv_gym_env.py, train_gym.py, usv_logic.py separata. | Compatibile con Stable-Baselines3 |

## Branch archiviati (tag)

Eliminati dalla lista branch ma recuperabili via `git checkout archive/<name>`:

| Tag | Descrizione |
|---|---|
| `archive/old` | Versione pre-refactor, identica a main |
| `archive/multi_maze_train` | Tentativo curriculum 50/50 Maze 1+2, abbandonato a metà |

---

## Analisi curriculum_learning — sintesi

Training completato: 3000 episodi su Maze 1 (Phase 1) e Maze 2 (Phase 2, curriculum progressivo).

| Maze | Successi training | Successi test (30 ep, ε=0) |
|---|---|---|
| Maze 1 | 54.6% (601/1100 ep) | **100%** (30/30) |
| Maze 2 | 4.2% (80/1900 ep) | **0%** (0/30) |
| Maze 3 (mai visto) | — | **0%** (0/30) |

**Cause root identificate:**
1. **Spawn fisso** → il robot memorizza il percorso specifico, non impara ad evitare ostacoli
2. **β=0.995** (ε decay troppo aggressivo) → esplorazione finisce a ~600 ep; Phase 2 inizia con ε≈0.08
3. **Reward complessa** → space_bonus e front_danger calibrati per Maze 1 aperto; in Maze 2 (81% muri diagonali) producono segnali fuorvianti
4. **Distributional shift** → Phase 1 solo muri axis-aligned; Maze 2 ha 25/31 muri diagonali, pattern LIDAR mai visti

Documentazione completa: `risultati/SESSIONE_ANALISI.md`

---

## Prossimo step pianificato: branch `random_spawn`

Approccio ispirato al paper originale (Feng et al., Robotics 2021):

| Parametro | curriculum_learning | random_spawn (proposto) |
|---|---|---|
| Reward | Complessa (5 componenti) | `+5 / −1000` puro |
| Spawn | Fisso | **Random** da lista precompilata |
| β (ε decay) | 0.995 | **0.999** |
| Maze | Curriculum Phase1→Phase2 | Maze 1 + Maze 2, random 50/50 |
| Base | — | Da `fast_sim` |

---

## Struttura file chiave

```
src/my_usv/
  scripts/
    usv_env.py       — ambiente ROS2 + reward + LIDAR processing
    ddqn_model.py    — rete neurale 50→300→300→11
    train.py         — loop training DDQN
    test.py          — valutazione ε=0
    patch_world.py   — scala velocità simulazione Gazebo
  worlds/
    labirinto_9a.world  — Maze 1: training (8×8m, muri axis-aligned)
    labirinto_9b.world  — Maze 2: training (15×13m, 81% muri diagonali)
    labirinto_10.world  — Maze 3: test only (mai visto in training)
risultati/
  analisi.py           — script analisi CSV + generazione plot
  SESSIONE_ANALISI.md  — documentazione completa analisi
  REPO_STATUS.md       — questo file
```

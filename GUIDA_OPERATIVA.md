# Guida Operativa — Branch `paper_implementation`

Questo branch contiene il tentativo di **curriculum learning** (nostra aggiunta al paper Feng 2021):  
Phase 1 su Maze 1 (semplice) → Phase 2 alternanza Maze 1/2.  
Il training da 6000 episodi eseguito su questo branch **non ha avuto successo** (crash rate >85%  
su entrambi i maze). Vedi `docs/report_paper_implementation.md` per l'analisi completa.

Il branch **`feng_direct`** implementa il metodo originale del paper (training diretto, no curriculum)
ed è il branch attivo per i nuovi esperimenti.

---

## Prerequisiti

1. **Docker Desktop** — avvialo e attendi l'icona verde "Engine Running".
2. **colcon build** — eseguire **una sola volta** (o dopo modifiche a CMakeLists/package.xml):
   ```bash
   docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
       bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
   ```

---

## Script disponibili

### `start_training_curriculum.sh`
Orchestratore del curriculum learning. Gestisce il ciclo di vita Docker in autonomia.

```
Uso: ./start_training_curriculum.sh          # avvia o riprende dal checkpoint
     ./start_training_curriculum.sh --reset  # cancella tutto e ricomincia da zero
```

Cosa fa:
- **Phase 1** (ep 0–500): allena solo su Maze 1 (labirinto_9a, spawn fisso x=-3 y=-5).
- **Phase 2** (ep 501–3000): alterna Maze 1 (30%) e Maze 2 (70%), spawn random da 8 posizioni.
  - Transizione automatica quando `avg_reward_100 > PHASE2_THRESHOLD` (soglia su reward).
- Ogni blocco da 100 episodi: avvia Gazebo headless a 4×, esegue `train.py`, ferma Gazebo.
- `BETA_DECAY=0.999` → epsilon raggiunge 0.05 a ep ≈ 3000.
- Stato curriculum: `src/my_usv/scripts/curriculum_state.txt`.
- Fase attuale: `src/my_usv/scripts/phase.txt`.
- Log per blocco: `logs/block_N_maze_M.log`.
- Checkpoint: `src/my_usv/scripts/checkpoint.pkl`.

> **Nota:** La transizione Phase 1→2 si basa sull'avg reward, non sul success rate.
> Con spawn fisso e reward denso (+5/step) la soglia viene raggiunta anche con crash rate ~80%.
> Questo è stato identificato come causa principale del fallimento del training.

---

### `start_test.sh`
Valutazione headless della policy su tutti e 3 i maze (30 episodi ciascuno).  
Gestisce il ciclo Docker in autonomia.

```
Uso: ./start_test.sh
```

Cosa fa:
- Verifica esistenza di `best_ddqn_model.pth`.
- Per ogni maze (1, 2, 3): avvia Gazebo headless 3×, lancia `test.py`, ferma Gazebo.
- Report comparativo finale (crash%, avg reward, avg steps per maze).
- Output: `src/my_usv/scripts/test_results.csv`.
- Maze 3 = test set (mai visto in training, usato solo qui).

---

### `test_spawns.sh [maze_id]`
Valida tutti gli 8 spawn point per Maze 1 e/o Maze 2.

```
Uso: ./test_spawns.sh        # entrambi i maze
     ./test_spawns.sh 1      # solo Maze 1
     ./test_spawns.sh 2      # solo Maze 2
```

Per ogni spawn: avvia Gazebo headless, spawna il robot, legge il primo scan LIDAR.
- ✅ OK — distanza muro > 0.40m
- ⚠️ WARNING — distanza 0.25–0.40m
- ❌ COLLISION — dentro muro → rimuovere dalla lista
- ⏱️ TIMEOUT — Gazebo non ha risposto

---

### `start_training_curriculum_multip.sh`
Variante del curriculum con sequenza di maze configurabile.  
Usata in esperimenti precedenti; non la versione principale.

```
Uso: ./start_training_curriculum_multip.sh
```

---

## Flusso consigliato — Rieseguire il training curriculum

> **Attenzione:** il training precedente (6000 ep) è già stato eseguito con risultati negativi.
> Prima di rieseguire, considera di usare `feng_direct` con `start_train_direct.sh`.

```bash
# 1. (Prima volta) Build
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"

# 2. Avvia curriculum da zero (cancella checkpoint precedente)
./start_training_curriculum.sh --reset

# 3. (Dopo training) Valutazione
./start_test.sh
```

Per **riprendere** un training interrotto (checkpoint esistente):
```bash
./start_training_curriculum.sh
```

---

## Risultati training eseguito (maggio 2026)

| Maze | Crash% (training) | Crash% (test) | Note |
|------|-------------------|---------------|------|
| Maze 1 | 85.6% (Block 2) | 73.3% | Peggiorato rispetto a Block 0 |
| Maze 2 | 88.1% (Block 2) | 73.3% | Mai appreso |
| Maze 3 | — (mai visto) | 60.0% | Migliore dei tre (generalizzazione casuale) |

Cause radice identificate:
1. Block 0 eseguito con `BETA_DECAY=0.995` (versione di codice errata) — epsilon a 0.05 a ep ≈ 600.
2. `PHASE2_THRESHOLD` basato su avg reward, non su success rate → Phase 2 attivata con crash rate 80%.
3. Catastrophic forgetting: Maze 2 contamina il replay buffer, degrada la policy su Maze 1.
4. Reward denso (+5/step) con spawn fisso maschera la mancanza di apprendimento reale.
5. Reset epsilon alla Phase 2 ininfluente (eps già >0.5 al trigger).

---

## Risoluzione problemi

**Container già in uso**
```bash
docker rm -f usv_container
```

**Gazebo crashato — log**
```bash
# Nome log: logs/block_N_maze_M.log
tail -30 logs/block_0_maze_1.log
```

**`/gazebo/set_entity_state` non trovato (hang)**  
Plugin `gazebo_ros_state` mancante. Verificare world files in `src/my_usv/worlds/` e `install/`.

**Modifiche Python non visibili**  
Live via volume — nessun rebuild necessario.

---

## File di stato

| File | Contenuto |
|------|-----------|
| `src/my_usv/scripts/checkpoint.pkl` | Weights, optimizer, replay buffer, epsilon, step globale |
| `src/my_usv/scripts/best_ddqn_model.pth` | Modello con miglior reward medio |
| `src/my_usv/scripts/training_log.csv` | Log episodio (reward, epsilon, crash, loss) — 6115 righe |
| `src/my_usv/scripts/test_results.csv` | Risultati test (30 ep × 3 maze) |
| `src/my_usv/scripts/curriculum_state.txt` | Episodio corrente del curriculum |
| `src/my_usv/scripts/phase.txt` | Fase attuale (1 o 2) |
| `logs/block_N_maze_M.log` | Log Gazebo per blocco |
| `docs/report_paper_implementation.md` | Analisi completa del training fallito |

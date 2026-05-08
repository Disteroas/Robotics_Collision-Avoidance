# Guida Operativa — Branch `feng_direct`

Questo branch implementa **Feng et al. 2021** nella sua forma originale:  
training diretto su mappa complessa, spawn random per-episodio da 8 posizioni,  
`BETA_DECAY=0.999`, nessun curriculum. Il branch `paper_implementation` contiene invece  
il tentativo di curriculum learning (nostra aggiunta, non nel paper).

---

## Prerequisiti

Prima di ogni sessione:

1. **Docker Desktop** — avvialo e attendi l'icona verde "Engine Running".
2. **colcon build** — eseguire **una sola volta** dopo il clone (o dopo modifiche a CMakeLists/package.xml):
   ```bash
   docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
       bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
   ```
   Propaga i file world da `src/` a `install/`. Per sole modifiche Python: non serve.

3. **XLaunch (VcXsrv)** — solo se usi `start_test_gui.sh`:
   - Multiple Windows → Start no client → **spunta "Disable access control"** → Fine.

---

## Script disponibili

### `start_train_direct.sh [maze_id]`
Training diretto metodo Feng 2021. Gestisce il ciclo di vita Docker in autonomia.

```
Uso: ./start_train_direct.sh [maze_id]
     maze_id: 1 o 2 (default: 2)
```

Cosa fa:
- Cancella `checkpoint.pkl`, `training_log.csv`, `phase.txt` (parte da zero).
- Avvia Gazebo headless a 5× velocità (`GAZEBO_SPEED=5`).
- Patcha il world file per la velocità (via `patch_world.py`).
- Lancia `train.py --maze-id N --end-ep 3000`.
- Spawn random per-episodio tramite `/gazebo/set_entity_state` (richiede plugin in world).
- `BETA_DECAY=0.999` → epsilon raggiunge 0.05 a ep ≈ 3000.
- Log Gazebo: `logs/direct_maze_N.log`.
- Checkpoint: `src/my_usv/scripts/checkpoint.pkl` ogni 100 episodi.
- Modello migliore: `src/my_usv/scripts/best_ddqn_model.pth`.

Per interrompere e riprendere: `Ctrl+C` + rilancia lo stesso script.  
Il checkpoint viene caricato automaticamente se presente.

---

### `start_test.sh`
Valutazione headless della policy su tutti e 3 i maze (30 episodi ciascuno).  
Gestisce il ciclo Docker in autonomia, non richiede container pre-avviato.

```
Uso: ./start_test.sh
```

Cosa fa:
- Verifica esistenza di `best_ddqn_model.pth` (esce se non trovato).
- Per ogni maze (1, 2, 3):
  - Avvia Gazebo headless a 3×.
  - Lancia `test.py --maze-id N --episodes 30 --model best_ddqn_model.pth`.
  - Ferma il container.
- Stampa report comparativo finale (crash%, avg reward, avg steps).
- Output CSV: `src/my_usv/scripts/test_results.csv`.
- Maze 1 e 2 = training set; Maze 3 = test set (mai visto in training).

---

### `start_test_gui.sh [maze_id] [episodes]`
Test visuale con GUI Gazebo (ispezionare il comportamento del robot).  
Richiede **XLaunch** attivo su Windows.

```
Uso: ./start_test_gui.sh [maze_id] [episodes]
     maze_id : 1, 2, 3 (default: 2)
     episodes: (default: 5)
```

Cosa fa:
- Avvia Gazebo con interfaccia grafica (`gui:=true`) via X11 forwarding (`DISPLAY=host.docker.internal:0.0`).
- Attende 35s avvio GUI (più lento del headless).
- Lancia `test.py` col modello migliore.
- Velocità: real-time (no patching world) — comportamento visibile a velocità normale.
- Alla fine (o Ctrl+C): rimuove automaticamente il container (trap EXIT).
- Risultati: `src/my_usv/scripts/test_gui_results.csv`.

---

### `test_spawns.sh [maze_id]`
Valida tutti gli 8 spawn point per Maze 1 e/o Maze 2.  
Usare prima di modificare le liste spawn in `usv_env.py`.

```
Uso: ./test_spawns.sh        # entrambi i maze
     ./test_spawns.sh 1      # solo Maze 1
     ./test_spawns.sh 2      # solo Maze 2
```

Per ogni spawn: avvia Gazebo headless, spawna il robot, legge il primo scan LIDAR.
- ✅ OK — distanza muro > 0.40m
- ⚠️ WARNING — distanza 0.25–0.40m (accettabile ma vicino)
- ❌ COLLISION — dentro muro → rimuovere dalla lista
- ⏱️ TIMEOUT — Gazebo non ha risposto

---

### `start_training_curriculum.sh`
Curriculum learning (Maze 1 → Maze 2, 100 ep per blocco).  
Presente per confronto con `paper_implementation`; **non è il metodo principale** di questo branch.

```
Uso: ./start_training_curriculum.sh          # avvia o riprende
     ./start_training_curriculum.sh --reset  # ricomincia da zero
```

---

## Flusso consigliato — Training + Test

```bash
# 1. (Prima volta) Build
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"

# 2. Training diretto su Maze 2 (Feng 2021)
./start_train_direct.sh 2

# 3. (Dopo training) Valutazione headless su tutti i maze
./start_test.sh

# 4. (Opzionale) Ispezione visiva su Maze 2
./start_test_gui.sh 2 10
```

---

## Flusso — Solo test visivo (modello già trainato)

```bash
# 1. Avvia XLaunch con "Disable access control"

# 2. Lancia test GUI
./start_test_gui.sh 2 5

# 3. Guarda il robot in Gazebo. Ctrl+C per fermare.
```

---

## Risoluzione problemi

**"Errore: modello non trovato"**  
`best_ddqn_model.pth` non esiste. Avvia prima un training o copia il modello da un altro branch.

**Container già in uso (`usv_container`)**  
```bash
docker rm -f usv_container
```

**Gazebo crashato — log vuoto**  
```bash
docker logs usv_container 2>&1 | tail -30
```

**Test GUI: schermo nero / Gazebo non appare**  
XLaunch non attivo o "Disable access control" non spuntato. Riavvia XLaunch.

**`/gazebo/set_entity_state` non trovato (hang al reset)**  
Il plugin `gazebo_ros_state` manca nel world file. Verifica che `labirinto_9a.world`,  
`labirinto_9b.world`, `labirinto_10.world` contengano:
```xml
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
  <ros><namespace>/gazebo</namespace></ros>
  <update_rate>1.0</update_rate>
</plugin>
```
Poi ri-eseguire `colcon build --packages-select my_usv`.

**Modifiche Python non visibili nel container**  
Le modifiche Python sono live via volume bind — nessun rebuild necessario.  
Solo CMakeLists.txt / package.xml / URDF richiedono `colcon build`.

---

## File di stato

| File | Contenuto |
|------|-----------|
| `src/my_usv/scripts/checkpoint.pkl` | Weights Q-net + target, optimizer, replay buffer, epsilon, step globale |
| `src/my_usv/scripts/best_ddqn_model.pth` | Snapshot del modello con miglior reward medio |
| `src/my_usv/scripts/training_log.csv` | Log episodio per episodio (reward, epsilon, crash, loss) |
| `src/my_usv/scripts/test_results.csv` | Risultati test headless (30 ep × 3 maze) |
| `src/my_usv/scripts/test_gui_results.csv` | Risultati test GUI |
| `logs/direct_maze_N.log` | Log stdout Gazebo durante training |

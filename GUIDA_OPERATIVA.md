# Guida Operativa — Branch `main`

Branch di riferimento originale. Contiene lo stack base (Gazebo + ROS2 + DDQN)  
con workflow manuale a due terminali. Non include gli script di automazione avanzati  
dei branch successivi.

Per il training attivo usa **`feng_direct`** (`start_train_direct.sh`).  
Per il curriculum learning usa **`paper_implementation`** (`start_training_curriculum.sh`).

---

## Prerequisiti

Prima di ogni sessione:

1. **Docker Desktop** — avvialo e attendi l'icona verde "Engine Running".
2. **XLaunch (VcXsrv)** — per la GUI di Gazebo:
   - Multiple Windows → Start no client → **spunta "Disable access control"** → Fine.
3. **colcon build** — una sola volta (o dopo modifiche a CMakeLists/package.xml):
   ```bash
   docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
       bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
   ```

---

## Script disponibili

### `start_sim.sh [maze_id]`
Avvia Gazebo **con GUI** nel container. Il terminale rimane bloccato a gestire Gazebo.

```
Uso: ./start_sim.sh [maze_id]
     maze_id: 1, 2, 3 (default: 1)
```

Cosa fa:
- Monta la cartella del progetto come volume nel container.
- Lancia `ros2 launch my_usv spawn_robot.launch.py` con world e coordinate per il maze scelto.
- Spawn fisso: Maze 1 (x=-3, y=-5, yaw=1.57), Maze 2 (x=-6, y=0), Maze 3 (x=-2, y=-1).
- Apre finestra Gazebo su Windows tramite XLaunch (DISPLAY=host.docker.internal:0.0).
- Container rimane attivo finché non premi Ctrl+C.

---

### `start_sim_headless.sh [maze_id]`
Avvia Gazebo **senza GUI** (headless), in background. Usare per training senza video.

```
Uso: ./start_sim_headless.sh [maze_id]
     maze_id: 1, 2, 3 (default: 1)
```

Cosa fa:
- Come `start_sim.sh` ma con `gui:=false`.
- Container in background (`-d`), terminale libero subito.
- Nome container: `usv_container_headless`.

---

### `start_train.sh`
Entra nel container Gazebo GUI già attivo e lancia `train.py`.  
Prerequisito: `start_sim.sh` deve essere in esecuzione in un altro terminale.

```
Uso: ./start_train.sh
```

Cosa fa:
- `docker exec -it usv_container` con winpty su Windows Git Bash.
- Lancia `train.py` senza argomenti (usa i default: maze corrente, nessun checkpoint).
- Non gestisce checkpoint automaticamente — va passato manualmente se serve.

---

### `start_train_headless.sh`
Come `start_train.sh` ma per il container headless.  
Prerequisito: `start_sim_headless.sh` deve essere in esecuzione.

```
Uso: ./start_train_headless.sh
```

Cosa fa:
- `docker exec -it usv_container_headless` con winpty.
- Lancia `train.py` nel container headless.

---

### `start_test.sh`
Entra nel container già attivo e lancia `test.py` (greedy, epsilon=0.0).  
Prerequisito: un container con Gazebo in esecuzione.

```
Uso: ./start_test.sh
```

Cosa fa:
- Verifica che il container `usv_container` sia in esecuzione (`docker inspect --State.Running`).
- `docker exec -it` con auto-detect winpty per Git Bash su Windows.
- Lancia `test.py` (usa il modello default, nessun argomento).
- Non genera CSV automaticamente — output solo su terminale.

---

## Flusso consigliato — Training con GUI (ispezione visiva)

Aprire **due terminali Git Bash** nella cartella del progetto.

**Terminale 1 — Gazebo:**
```bash
./start_sim.sh 1
```
Lasciarlo aperto. Gazebo compare su Windows.

**Terminale 2 — Training:**
```bash
./start_train.sh
```
Il robot inizia a muoversi in Gazebo. I log scorrono nel terminale.

Per cambiare maze: Ctrl+C nel Terminale 1, poi `./start_sim.sh 2`, poi riavvia `start_train.sh`.

---

## Flusso consigliato — Training headless (veloce, senza video)

**Terminale unico:**
```bash
# Avvia Gazebo headless (background)
./start_sim_headless.sh 1

# Avvia training
./start_train_headless.sh
```

---

## Flusso consigliato — Test policy

```bash
# Prerequisito: Gazebo già aperto con start_sim.sh o start_sim_headless.sh
./start_test.sh
```

---

## Modifica del codice

Le modifiche ai file Python (`train.py`, `usv_env.py`, ecc.) sono **live** via volume bind.  
Non serve rebuild. Per applicare: `Ctrl+C` su `start_train.sh`, poi rilanciarlo.

Solo questi file richiedono `colcon build`:
- `CMakeLists.txt`
- `package.xml`
- File URDF/SDF/world (propagazione da `src/` a `install/`)

---

## Risoluzione problemi

**"Container name already in use"**
```bash
docker rm -f usv_container
docker rm -f usv_container_headless
```

**`start_test.sh` non risponde (robot fermo)**  
Container non in esecuzione. Avvia prima `start_sim.sh` o `start_sim_headless.sh`.

**Schermo nero / Gazebo non appare**  
XLaunch non attivo o "Disable access control" non spuntato. Riavvia XLaunch.

**"Package my_usv not found"**  
Esegui `colcon build`:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
```

---

## Struttura branch

| Branch | Scopo |
|--------|-------|
| `main` | Stack base, workflow manuale (questo branch) |
| `paper_implementation` | Curriculum learning (Phase 1→2), esperimento completato |
| `feng_direct` | Metodo Feng 2021 originale (training diretto, branch attivo) |

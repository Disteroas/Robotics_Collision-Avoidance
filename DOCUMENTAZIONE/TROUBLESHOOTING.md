# Troubleshooting — problemi noti e soluzioni

---

## Container e Docker

**Container già in uso**
```bash
docker rm -f usv_container
```

**`docker run` fallisce — "engine not running"**  
Docker Desktop non è avviato. Aprilo e attendi l'icona verde "Engine Running".

**Log Gazebo**
```bash
docker logs usv_container 2>&1 | tail -50
```

---

## Gazebo

**Hang al reset — `/gazebo/set_entity_state` non trovato**  
Il plugin `gazebo_ros_state` manca nel world file. Verificare che ogni `.world` contenga:
```xml
<plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
  <ros><namespace>/gazebo</namespace></ros>
  <update_rate>1.0</update_rate>
</plugin>
```
Poi: `colcon build --packages-select my_usv`.  
Nota: `labirinto_10.world` (Maze 3) aveva questo problema — fixato nel branch `feng_direct`.

**Gazebo crasha subito (log vuoto)**  
Può essere un conflitto con un container precedente. Fare `docker rm -f usv_container` e riprovare.

**Simulazione molto lenta**  
Controllare che `patch_world.py` abbia applicato `real_time_update_rate`. Lo script di training lo chiama automaticamente.

---

## Training

**"Modello non trovato: best_ddqn_model.pth"**  
`best_ddqn_model.pth` non esiste. Avvia prima un training, o copia il modello da un altro branch.

**Checkpoint non caricato (training riparte da ep 0)**  
`checkpoint.pkl` non trovato nel path atteso (`src/my_usv/scripts/`). Verificare path in `start_train_direct.sh`.

**Crash a step=1 frequenti (reward=-1000 a step 1)**  
Cause possibili:
1. Spawn in posizione non sicura (imprecisione Gazebo ±2-3cm). Il safety check in `reset_environment` riprova fino a 3 volte ma se tutti i retry falliscono, procede comunque. → Validare spawn list con `./test_spawns.sh 2`.
2. Bug test.py (fixato in commit `4bbc476`): `reset_environment()` senza `maze_id` → spawn da Maze 1 in mondo Maze 2.

**LIDAR spam nei log (riga INFO per ogni episodio)**  
`_lidar_checked` veniva resettato in `reset_environment`. Fixato nel branch `feng_direct` — il messaggio LIDAR appare solo una volta all'avvio.

**Modifiche Python non visibili nel container**  
Le modifiche sono live via volume bind. Se sembra che il container usi codice vecchio, verificare che il file sia salvato sul host e che il path sia corretto.

---

## Test GUI (Gazebo con interfaccia grafica)

**Schermo nero / Gazebo non appare**  
XLaunch non attivo o "Disable access control" non spuntato. Riavviare XLaunch:
- Multiple Windows → Start no client → **spunta "Disable access control"** → Fine.

**Test GUI lento (robot si muove a rallentatore)**  
Normale: il test GUI usa real-time (no patching world). Se serve velocità accelerata, usare `start_test.sh` (headless).

---

## Build

**`colcon build` non necessario per modifiche Python**  
Solo CMakeLists.txt / package.xml / URDF / world files richiedono rebuild. Python è live via volume bind.

**`colcon build` fallisce con errori CMake**  
Verificare di essere dentro il container:
```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
```

---

## Test e validazione spawn

**`test_spawns.sh` mostra ❌ COLLISION su uno spawn**  
Lo spawn è dentro o troppo vicino a un muro. Rimuoverlo da `SPAWN_LISTS` in `usv_env.py` e dalla lista `SPAWNS_2` in `test_spawns.sh` (le due liste sono indipendenti — tenerle sincronizzate manualmente).

**`test_spawns.sh` mostra risultati diversi tra due run**  
Gazebo ha variabilità ±2-3cm nel posizionamento. Spawn con min_lidar 0.40-0.45m possono passare/fallire in modo non deterministico. Spostare gli spawn borderline di almeno 0.10m dalla posizione corrente.

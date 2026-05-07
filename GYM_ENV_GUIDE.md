# Guida Training — branch `gym_env`

**Branch:** `gym_env` (basato su `paper_implementation` features + wrapper Gymnasium)

Questo branch estende `paper_implementation` aggiungendo un wrapper Gymnasium attorno a `UsvEnv`. Questo permette di usare qualsiasi algoritmo compatibile con Gymnasium (Stable-Baselines3, CleanRL, XinJingHao DRL) semplicemente cambiando l'import in `train_gym.py`.

---

## Prerequisiti

1. **Docker Desktop** in esecuzione (icona verde)
2. **colcon build** già eseguito almeno una volta (necessario dopo aver clonato o dopo modifiche a CMakeLists.txt/world files):

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
  bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
```

> **Importante:** il world file include ora il plugin `gazebo_ros_state` che espone il servizio `/gazebo/set_entity_state`. Questo è necessario per lo spawn casuale per episodio. Se non si esegue `colcon build`, Gazebo non troverà il service e il training si bloccherà in attesa.

---

## Due modalità di training

### Modalità A — Curriculum DDQN (raccomandato)

Usa `train.py` + `usv_env.py` direttamente. Curriculum Phase 1 (Maze 1) → Phase 2 (70% Maze 2 + 30% Maze 1). Identico a `paper_implementation`.

```bash
./start_training_curriculum.sh          # avvia o riprende
./start_training_curriculum.sh --reset  # ricomincia da zero
```

**Parametri chiave:**

| Parametro | Valore |
|---|---|
| Episodi totali | 3000 |
| Episodi per blocco | 100 |
| Velocità Gazebo | 4× |
| Soglia Phase 2 | avg50_maze1 > 1500 |
| ε decay β | 0.999 (raggiunge 0.05 a ep 3000) |
| ε reset Phase 2 | max(ε, 0.5) — Narvekar et al. 2020 |
| Spawn | Random per-episode da `SPAWN_LISTS` in `usv_env.py` |
| Reward | +5 per step / −1000 collisione (Feng et al. 2021) |

**Output:**
- `src/my_usv/scripts/checkpoint.pkl` — checkpoint completo
- `src/my_usv/scripts/best_ddqn_model.pth` — miglior modello per avg100
- `src/my_usv/scripts/training_log.csv` — log episodio per episodio
- `logs/block_N_maze_M.log` — log Gazebo per ogni blocco

---

### Modalità B — Gymnasium single-maze

Usa `train_gym.py` + `usv_gym_env.py`. Singolo maze, nessun curriculum. Serve per testare algoritmi alternativi via swap point.

**Avvio manuale** (due terminali):

**Terminale 1 — Gazebo:**
```bash
docker run -d --name usv_container \
  --volume="/$(pwd):/home/usv_ws" \
  usv_rl_project \
  bash -c "
    cd /home/usv_ws && source install/setup.bash &&
    python3 src/my_usv/scripts/patch_world.py \
      /home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world \
      4 /tmp/world_fast.world &&
    ros2 launch my_usv spawn_robot.launch.py \
      world:=/tmp/world_fast.world x:=-3 y:=-5 yaw:=1.57 gui:=false
  "

sleep 30
```

**Terminale 2 — Training:**
```bash
docker exec usv_container bash -c "
  cd /home/usv_ws && source install/setup.bash &&
  python3 src/my_usv/scripts/train_gym.py \
    --maze-id 1 \
    --episodes 3000 \
    --max-steps 1000 \
    --checkpoint src/my_usv/scripts/checkpoint_gym.pth
"
```

**Output:**
- `src/my_usv/scripts/checkpoint_gym.pth` — checkpoint (formato diverso da `checkpoint.pkl`)
- `src/my_usv/scripts/best_gym_model.pth` — miglior modello per avg100

**Swap algoritmo** (in `train_gym.py`):
```python
# Sostituire questa riga:
from train_core import DDQNAgent

# Con uno di questi:
# from xjh_ddqn import DQN_Agent as DDQNAgent   # XinJingHao DDQN
# from xjh_ppo   import PPO_Agent as DDQNAgent   # XinJingHao PPO
```
L'agente deve esporre: `act(state)`, `memory.push(...)`, `learn()`, `step_done()`, `decay_epsilon()`.

---

## Differenze rispetto a `paper_implementation`

`gym_env` contiene tutte le modifiche di `paper_implementation` più:

| Aggiunta | File | Descrizione |
|---|---|---|
| Wrapper Gymnasium | `usv_gym_env.py` | Espone `reset()`, `step()`, `close()` standard |
| Training gymnasium | `train_gym.py` | Loop training con terminated/truncated separati |

`usv_gym_env.reset()` accetta `options={'maze_id': N}` e lo passa a `UsvEnv.reset_environment(maze_id=N)` per lo spawn casuale.

---

## Test unitari

```bash
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v"
```

Include `test_usv_gym_env.py` in aggiunta ai test standard.

---

## Troubleshooting

| Problema | Causa | Fix |
|---|---|---|
| `Attendo /gazebo/set_entity_state...` all'infinito | `colcon build` non eseguito dopo modifica world file | Eseguire `colcon build --packages-select my_usv` |
| `docker run fallito` al blocco 1 | Container precedente ancora attivo | `docker rm -f usv_container` |
| LIDAR sync failures | Gazebo speed > 4× | Mantenere `GAZEBO_SPEED=4` |
| Container bloccato | Training interrotto male | `docker rm -f usv_container` |

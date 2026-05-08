#!/bin/bash
# =============================================================================
#  start_train_direct.sh  –  Training diretto su Maze 2 (Feng et al. 2021)
#
#  Implementa il metodo del paper senza curriculum learning:
#  - Training diretto su Maze 2 (mappa complessa)
#  - Spawn random per-episodio da 8 posizioni predefinite
#  - BETA_DECAY=0.999 per 3000 episodi
#  - Reward: +5/step, -1000/collision
#
#  Uso: ./start_train_direct.sh [maze_id]
#       maze_id: 1, 2 (default: 2)
#
#  PREREQUISITO: colcon build deve essere stato eseguito una volta:
#    docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
#      bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"
# =============================================================================

MAZE_ID=${1:-2}
GAZEBO_SPEED=5
GAZEBO_WAIT=30
EPISODES=3000

declare -A WORLD_PATH SPAWN
WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"

SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"

if [[ -z "${WORLD_PATH[$MAZE_ID]}" ]]; then
    echo "Errore: maze_id deve essere 1 o 2. Ricevuto: $MAZE_ID"
    exit 1
fi

PATCHED_WORLD="/tmp/world_fast.world"
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="${SCRIPTS_CTR}/checkpoint.pkl"
LOG_ID="direct_maze_${MAZE_ID}"

mkdir -p "$(pwd)/logs"

echo ""
echo "============================================================"
echo "  USV DDQN — TRAINING DIRETTO (Feng et al. 2021)"
echo "============================================================"
echo "  Maze        : ${MAZE_ID}"
echo "  Episodi     : ${EPISODES}"
echo "  BETA_DECAY  : 0.999"
echo "  Spawn       : random per-episodio (8 posizioni)"
echo "  Curriculum  : NESSUNO — training diretto"
echo "  Checkpoint  : src/my_usv/scripts/checkpoint.pkl"
echo "============================================================"
echo ""

# Rimuovi checkpoint precedente per partire da zero
if [ -f "src/my_usv/scripts/checkpoint.pkl" ]; then
    echo "Rimozione checkpoint precedente..."
    rm -f src/my_usv/scripts/checkpoint.pkl
fi
if [ -f "src/my_usv/scripts/training_log.csv" ]; then
    echo "Rimozione training_log precedente..."
    rm -f src/my_usv/scripts/training_log.csv
fi
if [ -f "src/my_usv/scripts/phase.txt" ]; then
    rm -f src/my_usv/scripts/phase.txt
fi

docker rm -f usv_container 2>/dev/null
sleep 1

echo "Avvio Gazebo (Maze ${MAZE_ID}, velocita' ${GAZEBO_SPEED}x headless)..."
LOG_FILE="$(pwd)/logs/${LOG_ID}.log"

docker run -d --rm --name usv_container \
    --volume="/$(pwd):/home/usv_ws" \
    usv_rl_project \
    bash -c "
        cd /home/usv_ws && \
        source install/setup.bash && \
        python3 ${SCRIPTS_CTR}/patch_world.py \
            '${WORLD_PATH[$MAZE_ID]}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
        ros2 launch my_usv spawn_robot.launch.py \
            world:=${PATCHED_WORLD} \
            ${SPAWN[$MAZE_ID]} \
            gui:=false
    " > "$LOG_FILE" 2>&1

if [ $? -ne 0 ]; then
    echo "Errore avvio container."
    exit 1
fi

echo "Attendo ${GAZEBO_WAIT}s avvio Gazebo..."
sleep "$GAZEBO_WAIT"

running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
if [ "$running" != "true" ]; then
    echo "Gazebo crashato. Log:"
    tail -20 "$LOG_FILE"
    exit 1
fi

echo "Gazebo OK. Avvio training su Maze ${MAZE_ID}..."
echo ""

docker exec usv_container \
    bash -c "
        cd /home/usv_ws && source install/setup.bash && \
        python3 ${SCRIPTS_CTR}/train.py \
            --maze-id   ${MAZE_ID} \
            --end-ep    ${EPISODES} \
            --checkpoint ${CHECKPOINT_CTR}
    "

echo ""
echo "Training completato."
docker rm -f usv_container 2>/dev/null

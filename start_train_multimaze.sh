#!/bin/bash
# =============================================================================
#  start_train_multimaze.sh — M2-only training (Sync & Robust)
# =============================================================================

GAZEBO_SPEED=5
GAZEBO_WAIT=35  # Leggermente aumentato per sicurezza
TOTAL_BLOCKS=20
BLOCK_SIZE=200
MAZE_ID=2

WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
SPAWN_ARGS="x:=-6 y:=0 yaw:=0"

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="${SCRIPTS_CTR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))

mkdir -p "$(pwd)/logs"

if [[ "$1" == "--reset" ]]; then
    echo "  [INFO] --reset: rimozione checkpoint e log precedenti..."
    rm -f src/my_usv/scripts/checkpoint.pkl
    rm -f src/my_usv/scripts/training_log.csv
    rm -f src/my_usv/scripts/best_ddqn_model.pth
    echo "  [INFO] Reset completato."
fi

trap 'echo -e "\n  [AVVISO] Interruzione ricevuta. Pulizia..."; docker rm -f usv_container &>/dev/null; exit 1' INT TERM

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Ep $((START_EP+1))-${END_EP}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container &>/dev/null
    sleep 2 # Tempo tecnico per pulizia socket

    LOG_FILE="$(pwd)/logs/training_block_$(printf '%02d' $b).log"

    # --shm-size=2gb fondamentale per evitare crash di Gazebo per mancanza RAM
    docker run -d --name usv_container \
        --shm-size=2gb \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/patch_world.py '${WORLD_PATH}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
            ros2 launch my_usv spawn_robot.launch.py world:=${PATCHED_WORLD} ${SPAWN_ARGS} gui:=false
        " > "$LOG_FILE" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "  [ERRORE] Avvio container fallito."
        exit 1
    fi

    echo "  Attendo ${GAZEBO_WAIT}s avvio Gazebo..."
    sleep "$GAZEBO_WAIT"

    # Controllo che il processo ROS sia vivo
    if [[ "$(docker inspect -f '{{.State.Running}}' usv_container)" != "true" ]]; then
        echo "  [ERRORE] Gazebo crashato! Controlla il log: $LOG_FILE"
        exit 1
    fi

    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/train.py \
                --maze-id    ${MAZE_ID} \
                --start-ep   ${START_EP} \
                --end-ep     ${END_EP} \
                --total-ep   ${TOTAL_EP} \
                --checkpoint ${CHECKPOINT_CTR}
        "
    
    EXIT_CODE=$?
    docker rm -f usv_container &>/dev/null

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  [ERRORE] train.py fallito al blocco ${b}. Codice: ${EXIT_CODE}"
        exit 1
    fi
    echo "  Blocco ${b} completato."
done

echo "Training completato."

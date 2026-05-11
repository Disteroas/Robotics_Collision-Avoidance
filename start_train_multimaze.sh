#!/bin/bash
# =============================================================================
#  start_train_multimaze.sh  —  Multi-maze interleaved training (M1:M2 = 1:2)
#
#  5000 episodi, 25 blocchi x 200 ep, pattern (M1, M2, M2) ciclico.
#  Gazebo riavvia ogni blocco per caricare il world file del maze corrente.
#  Checkpoint condiviso: epsilon e replay buffer continuano senza reset.
#
#  Uso:
#    ./start_train_multimaze.sh           # riprende da checkpoint esistente
#    ./start_train_multimaze.sh --reset   # cancella tutto e riparte da zero
#
#  PREREQUISITO: colcon build eseguito almeno una volta.
# =============================================================================

GAZEBO_SPEED=5
GAZEBO_WAIT=30
TOTAL_BLOCKS=25
BLOCK_SIZE=200
BLOCK_PATTERN=(1 2 2)

WORLD_PATH_1="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH_2="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
SPAWN_ARGS_1="x:=-3 y:=-5 yaw:=1.57"
SPAWN_ARGS_2="x:=-6 y:=0 yaw:=0"

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="${SCRIPTS_CTR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))

mkdir -p "$(pwd)/logs"

if [[ "$1" == "--reset" ]]; then
    echo "  --reset: rimozione checkpoint e log precedenti..."
    rm -f src/my_usv/scripts/checkpoint.pkl
    rm -f src/my_usv/scripts/training_log.csv
    rm -f src/my_usv/scripts/best_ddqn_model.pth
    rm -f src/my_usv/scripts/phase.txt
    echo "  Reset completato."
fi

echo ""
echo "============================================================"
echo "  USV DDQN — MULTI-MAZE INTERLEAVED TRAINING"
echo "============================================================"
echo "  Maze pattern : M1/M2/M2 (ratio 1:2)"
echo "  Episodi tot  : ${TOTAL_EP}"
echo "  Blocchi      : ${TOTAL_BLOCKS} x ${BLOCK_SIZE} ep"
echo "  Gazebo speed : ${GAZEBO_SPEED}x headless"
echo "  Checkpoint   : src/my_usv/scripts/checkpoint.pkl"
echo "============================================================"
echo ""

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    pattern_idx=$(( (b - 1) % ${#BLOCK_PATTERN[@]} ))
    MAZE_ID=${BLOCK_PATTERN[$pattern_idx]}

    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    if [[ $MAZE_ID -eq 1 ]]; then
        WORLD_PATH="$WORLD_PATH_1"
        SPAWN_ARGS="$SPAWN_ARGS_1"
    else
        WORLD_PATH="$WORLD_PATH_2"
        SPAWN_ARGS="$SPAWN_ARGS_2"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Maze ${MAZE_ID} | ep $((START_EP+1))-${END_EP}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container 2>/dev/null
    sleep 1

    LOG_FILE="$(pwd)/logs/multimaze_block_$(printf '%02d' $b)_maze_${MAZE_ID}.log"

    docker run -d --rm --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/patch_world.py \
                '${WORLD_PATH}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
            ros2 launch my_usv spawn_robot.launch.py \
                world:=${PATCHED_WORLD} \
                ${SPAWN_ARGS} \
                gui:=false
        " > "$LOG_FILE" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "  ERRORE: avvio container fallito al blocco ${b}."
        exit 1
    fi

    echo "  Attendo ${GAZEBO_WAIT}s avvio Gazebo..."
    sleep "$GAZEBO_WAIT"

    running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
    if [[ "$running" != "true" ]]; then
        echo "  ERRORE: Gazebo crashato al blocco ${b}. Ultimi log:"
        tail -20 "$LOG_FILE"
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
    docker rm -f usv_container 2>/dev/null

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  ERRORE: train.py exit ${EXIT_CODE} al blocco ${b}. Interruzione."
        exit 1
    fi

    echo "  Blocco ${b}/${TOTAL_BLOCKS} completato."
done

echo ""
echo "============================================================"
echo "  TRAINING COMPLETATO — ${TOTAL_EP} episodi"
echo "============================================================"
echo "  Modello: src/my_usv/scripts/best_ddqn_model.pth"
echo "  Log:     src/my_usv/scripts/training_log.csv"
echo "  Blocchi: logs/multimaze_block_*.log"
echo "============================================================"

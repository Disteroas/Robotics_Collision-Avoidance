#!/bin/bash
# =============================================================================
#  start_train_multimaze.sh  —  M2-only training
#
#  Training esclusivo sul Labirinto 2 (Maze 2) con 4000 episodi totali,
#  divisi in 20 blocchi da 200 episodi per prevenire memory leak di Gazebo.
#
#  Uso:
#    ./start_train_multimaze.sh           # riprende da checkpoint esistente
#    ./start_train_multimaze.sh --reset   # cancella tutto e riparte da zero
# =============================================================================

GAZEBO_SPEED=5
GAZEBO_WAIT=30
TOTAL_BLOCKS=20
BLOCK_SIZE=200
MAZE_ID=2

WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
# Punto di inserimento iniziale sicuro in Gazebo (il random spawn vero avviene via Python)
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

echo ""
echo "============================================================"
echo "  USV DDQN — M2-ONLY TRAINING"
echo "============================================================"
echo "  Target       : Maze 2 (Labirinto 9b)"
echo "  Episodi tot  : ${TOTAL_EP} (${TOTAL_BLOCKS} blocchi x ${BLOCK_SIZE})"
echo "  Gazebo speed : ${GAZEBO_SPEED}x headless"
echo "============================================================"
echo ""

trap 'echo -e "\n  [AVVISO] Interruzione ricevuta. Pulizia container..."; docker rm -f usv_container &>/dev/null; exit 1' INT TERM

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Maze ${MAZE_ID} | ep $((START_EP+1))-${END_EP}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container &>/dev/null
    sleep 1

    LOG_FILE="$(pwd)/logs/training_block_$(printf '%02d' $b)_maze_${MAZE_ID}.log"

    docker run -d --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/patch_world.py '${WORLD_PATH}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
            ros2 launch my_usv spawn_robot.launch.py world:=${PATCHED_WORLD} ${SPAWN_ARGS} gui:=false
        " > "$LOG_FILE" 2>&1

    if [[ $? -ne 0 ]]; then
        echo "  [ERRORE] Avvio container fallito al blocco ${b}."
        exit 1
    fi

    echo "  Attendo ${GAZEBO_WAIT}s avvio Gazebo..."
    sleep "$GAZEBO_WAIT"

    running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
    if [[ "$running" != "true" ]]; then
        echo "  [ERRORE] Gazebo crashato al blocco ${b}. Ultimi log:"
        tail -10 "$LOG_FILE"
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
        echo "  [ERRORE] train.py terminato con codice ${EXIT_CODE} al blocco ${b}."
        exit 1
    fi
    echo "  Blocco ${b} completato con successo."
done

echo ""
echo "============================================================"
echo "  TRAINING COMPLETATO — ${TOTAL_EP} episodi"
echo "============================================================"
echo "  Modello salvato in: src/my_usv/scripts/best_ddqn_model.pth"

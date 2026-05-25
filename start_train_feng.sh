#!/bin/bash
# =============================================================================
#  start_train_feng.sh  —  Replica fedele Feng et al. 2021 (baseline)
#
#  Training DIRETTO su Maze 2 (no curriculum, no multi-maze), 3000 episodi,
#  spawn random per-episodio. Agente Feng-puro (stato 50 [0,5], reward +5/-1000,
#  no frame-stack/heading/DR/grad-clip). Gazebo riavvia ogni blocco.
#  Eval invariata: usare ./start_test.sh --config=feng --seed=N.
#
#  Uso:
#    ./start_train_feng.sh --seed=0 --config=feng           # riprende
#    ./start_train_feng.sh --seed=0 --config=feng --reset   # backup + riparta
#
#  PREREQUISITO: colcon build eseguito almeno una volta.
# =============================================================================

GAZEBO_SPEED=5
GAZEBO_WAIT=30
TOTAL_BLOCKS=15      # 3000 ep = 15 × 200 (Feng: 3000 epoch)
BLOCK_SIZE=200
MAZE_ID=2            # Feng: training su una sola mappa complessa (≈ Map 2)

SEED=0
CONFIG="feng"
DO_RESET=0
for arg in "$@"; do
    case "$arg" in
        --reset)    DO_RESET=1 ;;
        --seed=*)   SEED="${arg#*=}" ;;
        --config=*) CONFIG="${arg#*=}" ;;
    esac
done
RUN_DIR="runs/${CONFIG}/seed_${SEED}"
mkdir -p "$(pwd)/${RUN_DIR}"

WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
SPAWN_ARGS="x:=-6 y:=0 yaw:=0"   # spawn di launch iniziale; per-episodio è random nell'env

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="/home/usv_ws/${RUN_DIR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))

mkdir -p "$(pwd)/logs"

if [[ "$DO_RESET" == "1" ]]; then
    BACKUP_DIR="ANALISI_TRAINING/$(date +%Y_%m_%d)/pre_reset_${CONFIG}_seed_${SEED}"
    if [[ -d "${RUN_DIR}" ]]; then
        echo "  --reset: backup di ${RUN_DIR} → ${BACKUP_DIR}"
        mkdir -p "${BACKUP_DIR}"
        if ! cp -r "${RUN_DIR}/." "${BACKUP_DIR}/"; then
            echo "  ❌ Backup fallito. Reset abortito per non perdere dati."
            exit 1
        fi
    fi
    echo "  --reset: rimozione artefatti in ${RUN_DIR}..."
    rm -f "${RUN_DIR}/checkpoint.pkl" "${RUN_DIR}/training_log.csv" \
          "${RUN_DIR}/best_model.pth"
    echo "  Reset completato."
fi

echo ""
echo "============================================================"
echo "  USV DDQN — REPLICA FEDELE FENG 2021 (baseline)"
echo "============================================================"
echo "  Maze         : M2 only (random spawn per-episodio)"
echo "  Episodi tot  : ${TOTAL_EP}"
echo "  Blocchi      : ${TOTAL_BLOCKS} x ${BLOCK_SIZE} ep"
echo "  Gazebo speed : ${GAZEBO_SPEED}x headless"
echo "  Seed/Config  : ${SEED} / ${CONFIG}"
echo "  Checkpoint   : ${RUN_DIR}/checkpoint.pkl"
echo "============================================================"
echo ""

trap 'echo ""; echo "  Interruzione ricevuta. Pulizia container..."; docker rm -f usv_container 2>/dev/null; exit 1' INT TERM

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    echo ""
    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Maze ${MAZE_ID} | ep $((START_EP+1))-${END_EP}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container 2>/dev/null
    sleep 1

    LOG_FILE="$(pwd)/logs/feng_block_$(printf '%02d' $b)_maze_${MAZE_ID}.log"

    docker run -d --name usv_container \
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
                --checkpoint ${CHECKPOINT_CTR} \
                --seed       ${SEED}
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
echo "  TRAINING COMPLETATO — ${TOTAL_EP} episodi (Feng baseline)"
echo "============================================================"
echo "  Modello: ${RUN_DIR}/best_model.pth"
echo "  Log:     ${RUN_DIR}/training_log.csv"
echo "============================================================"

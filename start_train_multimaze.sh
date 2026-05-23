#!/bin/bash
# =============================================================================
#  start_train_multimaze.sh — M2-only training (Sync & Robust, multi-seed)
#
#  Uso:
#    ./start_train_multimaze.sh           → seed 42 (default)
#    ./start_train_multimaze.sh --seed 1  → seed 1
#    ./start_train_multimaze.sh --reset   → cancella checkpoint, poi seed 42
#    ./start_train_multimaze.sh --reset --seed 7 → reset + seed 7
#
#  Per run multi-seed (min 3 seed per config):
#    ./start_train_multimaze.sh --seed 42
#    ./start_train_multimaze.sh --seed 1
#    ./start_train_multimaze.sh --seed 7
# =============================================================================

GAZEBO_SPEED=3
GAZEBO_WAIT=35

# FIX: era TOTAL_BLOCKS=20 BLOCK_SIZE=200 → 20 cold restart di Gazebo.
# I dati di training mostravano un crollo del crash rate da 86% → 99% esattamente
# al blocco 18 (ep 3600): classico catastrophic forgetting da cold restart.
# Con 8 blocchi da 500 ep si ha lo stesso totale (4000 ep) ma solo 8 restart,
# riducendo drasticamente l'instabilità da reinizializzazione Gazebo.
TOTAL_BLOCKS=8
BLOCK_SIZE=500

MAZE_ID=2
SEED=42

WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
SPAWN_ARGS="x:=-6 y:=0 yaw:=0"

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="${SCRIPTS_CTR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"
TOTAL_EP=$(( TOTAL_BLOCKS * BLOCK_SIZE ))

mkdir -p "$(pwd)/logs"

# Parsing argomenti
DO_RESET=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reset) DO_RESET=1; shift ;;
        --seed)  SEED="$2"; shift 2 ;;
        *) echo "Argomento sconosciuto: $1"; exit 1 ;;
    esac
done

if [[ "$DO_RESET" -eq 1 ]]; then
    echo "  [INFO] --reset: backup CSV e rimozione checkpoint..."
    TS=$(date +%Y%m%d_%H%M%S)
    mkdir -p src/my_usv/scripts/ANALISI_BACKUP
    [ -f src/my_usv/scripts/training_log.csv ] && \
        cp src/my_usv/scripts/training_log.csv \
           "src/my_usv/scripts/ANALISI_BACKUP/training_log_${TS}_pre_reset.csv" && \
        echo "  [BACKUP] training_log.csv → ANALISI_BACKUP/"
    rm -f src/my_usv/scripts/checkpoint.pkl
    rm -f src/my_usv/scripts/training_log.csv
    rm -f src/my_usv/scripts/best_ddqn_model.pth
    echo "  [INFO] Reset completato. Seed=$SEED"
fi

echo "  [INFO] Avvio training — SEED=${SEED} | TOTAL_EP=${TOTAL_EP} | BLOCCHI=${TOTAL_BLOCKS}×${BLOCK_SIZE} | MAZE_ID=${MAZE_ID}"

trap 'echo -e "\n  [AVVISO] Interruzione ricevuta. Pulizia..."; docker rm -f usv_container &>/dev/null; exit 1' INT TERM

for (( b=1; b<=TOTAL_BLOCKS; b++ )); do
    START_EP=$(( (b - 1) * BLOCK_SIZE ))
    END_EP=$(( b * BLOCK_SIZE ))

    echo "------------------------------------------------------------"
    echo "  Blocco ${b}/${TOTAL_BLOCKS} | Ep $((START_EP+1))-${END_EP} | seed=${SEED}"
    echo "------------------------------------------------------------"

    docker rm -f usv_container &>/dev/null
    sleep 2

    LOG_FILE="$(pwd)/logs/training_block_$(printf '%02d' $b)_seed${SEED}.log"

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

    if [[ "$(docker inspect -f '{{.State.Running}}' usv_container)" != "true" ]]; then
        echo "  [ERRORE] Gazebo crashato! Controlla: $LOG_FILE"
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
    docker rm -f usv_container &>/dev/null

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "  [ERRORE] train.py fallito al blocco ${b}. Codice: ${EXIT_CODE}"
        exit 1
    fi
    echo "  Blocco ${b} completato."
done

echo "  Training completato. Seed=${SEED}"

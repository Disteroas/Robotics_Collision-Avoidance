#!/bin/bash
# =============================================================================
#  start_training_curriculum.sh  –  Orchestratore del curriculum multi-labirinto.
#
#  Questo script sostituisce start_train.sh quando si vuole allenare su
#  più labirinti. Gestisce l'intero ciclo di vita Docker da solo.
#  MODIFICATO: Supporto Multi-Piattaforma (Linux/Windows) e GUI Attiva.
# =============================================================================

# --------------------------------------------------------------------------- #
#  CONFIGURAZIONE CURRICULUM  ← modifica questi valori                         #
# --------------------------------------------------------------------------- #
TOTAL_EPISODES=3000          # episodi totali di training
EPISODES_PER_BLOCK=100       # episodi per blocco prima di cambiare labirinto
GAZEBO_SPEED=5               # real_time_factor (5 = 5x più veloce del real-time)
GAZEBO_STARTUP_WAIT=30       # secondi da aspettare (alzato a 30s per caricare la GUI)

# Sequenza di labirinti
MAZE_SEQUENCE=(1 2 3)

# Percorsi interni al container
SCRIPTS_DIR_CONTAINER="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CONTAINER="${SCRIPTS_DIR_CONTAINER}/checkpoint.pkl"

# Percorsi sul host
SCRIPTS_DIR_HOST="$(pwd)/src/my_usv/scripts"
STATE_FILE="${SCRIPTS_DIR_HOST}/curriculum_state.txt"

# Directory per i log di Gazebo
mkdir -p "$(pwd)/logs"

# --------------------------------------------------------------------------- #
#  RILEVAMENTO OS E SETUP GRAFICA (MULTI-PLATFORM)                            #
# --------------------------------------------------------------------------- #
OS_TYPE=$(uname -s)
DOCKER_GUI_ARGS=""

if [[ "$OS_TYPE" == "Linux" ]] && [[ ! $(uname -r) =~ "microsoft" ]]; then
    # Sistema: Linux Nativo
    echo "🖥️  Rilevato sistema Linux. Sblocco xhost per la grafica..."
    xhost +local:root &> /dev/null
    DOCKER_GUI_ARGS="--env=DISPLAY --env=QT_X11_NO_MITSHM=1 --volume=$HOME/.Xauthority:/root/.Xauthority:rw --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --device=/dev/dri:/dev/dri --group-add video"
else
    # Sistema: Windows (WSL) o Mac
    echo "🖥️  Rilevato sistema Windows/Mac. Imposto grafica per VcXsrv/XQuartz..."
    DOCKER_GUI_ARGS="--env=DISPLAY=host.docker.internal:0.0 --env=QT_X11_NO_MITSHM=1 --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw"
fi

# --------------------------------------------------------------------------- #
#  CONFIGURAZIONE LABIRINTI                                                   #
# --------------------------------------------------------------------------- #
declare -A WORLD_PATHS
WORLD_PATHS[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATHS[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATHS[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"

declare -A SPAWN_COORDS
SPAWN_COORDS[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN_COORDS[2]="x:=-6 y:=0 yaw:=0"
SPAWN_COORDS[3]="x:=-2 y:=-1 yaw:=0"

declare -A MAZE_NAMES
MAZE_NAMES[1]="Labirinto 1  (labirinto_9a)"
MAZE_NAMES[2]="Labirinto 2  (labirinto_9b)"
MAZE_NAMES[3]="Labirinto 3  (labirinto_10)"

# --------------------------------------------------------------------------- #
#  CALCOLI DERIVATI E RESET FLAG                                              #
# --------------------------------------------------------------------------- #
TOTAL_BLOCKS=$(( (TOTAL_EPISODES + EPISODES_PER_BLOCK - 1) / EPISODES_PER_BLOCK ))
NUM_MAZES=${#MAZE_SEQUENCE[@]}

if [ "$1" = "--reset" ]; then
    echo ""
    echo "⚠️  Flag --reset: cancello checkpoint e stato curriculum."
    read -r -p "   Sei sicuro? (y/N) " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -f "${SCRIPTS_DIR_HOST}/checkpoint.pkl" \
              "${SCRIPTS_DIR_HOST}/checkpoint.pkl.tmp" \
              "${STATE_FILE}"
        echo "   ✅ Reset completato. Ripartenza da zero."
    else
        echo "   ❌ Annullato."
        exit 0
    fi
    echo ""
fi

# --------------------------------------------------------------------------- #
#  FUNZIONI                                                                   #
# --------------------------------------------------------------------------- #

cleanup_container() {
    if docker inspect -f '{{.State.Running}}' usv_container &>/dev/null; then
        echo "  [Docker] Container 'usv_container' già attivo, lo termino..."
        docker stop usv_container &>/dev/null || true
        sleep 3
    fi
}

start_gazebo() {
    local maze_id=$1
    local block_num=$2

    local world_path="${WORLD_PATHS[$maze_id]}"
    local coords="${SPAWN_COORDS[$maze_id]}"
    local log_file="$(pwd)/logs/sim_block_${block_num}.log"
    local patched_world="/tmp/world_fast.world"

    echo "  [Gazebo] Avvio CON GRAFICA ${MAZE_NAMES[$maze_id]} (${GAZEBO_SPEED}x)..."
    echo "  [Gazebo] Log: logs/sim_block_${block_num}.log"

    # docker run con inserimento dinamico degli argomenti GUI ($DOCKER_GUI_ARGS)
    docker run -d --rm --name usv_container --net=host \
        $DOCKER_GUI_ARGS \
        --volume="$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_DIR_CONTAINER}/patch_world.py \
                '${world_path}' ${GAZEBO_SPEED} ${patched_world} && \
            ros2 launch my_usv spawn_robot.launch.py \
                world:=${patched_world} \
                ${coords} \
                gui:=true
        " > "$log_file" 2>&1

    local docker_exit=$?
    if [ $docker_exit -ne 0 ]; then
        echo "  ❌ ERRORE: docker run ha fallito (exit code $docker_exit)."
        return 1
    fi

    echo "  [Gazebo] Container avviato. Attendo ${GAZEBO_STARTUP_WAIT}s per caricare la grafica..."
    sleep "$GAZEBO_STARTUP_WAIT"

    if ! docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null | grep -q 'true'; then
        echo "  ❌ ERRORE: Gazebo è crashato durante l'avvio!"
        tail -30 "$log_file"
        return 1
    fi

    echo "  [Gazebo] ✅ Pronto."
    return 0
}

run_training_block() {
    local start_ep=$1
    local end_ep=$2
    local maze_id=$3
    local block_num=$4

    echo "  [Train] Lancio training: ep ${start_ep}+1 → ${end_ep} | ${MAZE_NAMES[$maze_id]}"

    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_DIR_CONTAINER}/train.py \
                --start-ep ${start_ep} \
                --end-ep   ${end_ep} \
                --maze-id  ${maze_id} \
                --checkpoint ${CHECKPOINT_CONTAINER}
        "
    return $?
}

stop_gazebo() {
    echo "  [Gazebo] Termino container..."
    docker stop usv_container &>/dev/null || true
    local waited=0
    while docker inspect usv_container &>/dev/null && [ $waited -lt 15 ]; do
        sleep 1
        ((waited++))
    done
    echo "  [Gazebo] ✅ Container terminato."
}

_global_interrupt=false
trap '_global_interrupt=true; echo ""; echo "⚠️  Ctrl+C ricevuto. Attendo fine blocco corrente..."; stop_gazebo; exit 0' INT TERM

# --------------------------------------------------------------------------- #
#  LOOP PRINCIPALE                                                            #
# --------------------------------------------------------------------------- #
START_BLOCK=0
if [ -f "$STATE_FILE" ]; then
    START_BLOCK=$(cat "$STATE_FILE")
    echo "📂 Curriculum state trovato: riprendo dal blocco $((START_BLOCK + 1))/$TOTAL_BLOCKS"
    echo ""
fi

total_start_time=$(date +%s)

for (( block=START_BLOCK; block<TOTAL_BLOCKS; block++ )); do

    BLOCK_START_EP=$(( block * EPISODES_PER_BLOCK ))
    BLOCK_END_EP=$(( BLOCK_START_EP + EPISODES_PER_BLOCK ))
    if [ $BLOCK_END_EP -gt $TOTAL_EPISODES ]; then
        BLOCK_END_EP=$TOTAL_EPISODES
    fi

    MAZE_IDX=$(( block % NUM_MAZES ))
    MAZE_ID=${MAZE_SEQUENCE[$MAZE_IDX]}

    elapsed=$(( $(date +%s) - total_start_time ))
    if [ $block -gt $START_BLOCK ] && [ $elapsed -gt 0 ]; then
        blocks_done=$(( block - START_BLOCK ))
        blocks_remaining=$(( TOTAL_BLOCKS - block ))
        sec_per_block=$(( elapsed / blocks_done ))
        eta_sec=$(( sec_per_block * blocks_remaining ))
        eta_str=$(printf '%dh %02dm' $((eta_sec/3600)) $((eta_sec%3600/60)))
    else
        eta_str="N/A"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  BLOCCO $((block + 1))/$TOTAL_BLOCKS  |  ${MAZE_NAMES[$MAZE_ID]}"
    echo "  Episodi: $((BLOCK_START_EP + 1)) → $BLOCK_END_EP  |  ETA: $eta_str"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cleanup_container

    if ! start_gazebo "$MAZE_ID" "$block"; then
        echo ""
        echo "❌ Gazebo non è partito. Salto e riprovo..."
        sleep 5
        continue
    fi

    run_training_block "$BLOCK_START_EP" "$BLOCK_END_EP" "$MAZE_ID" "$block"
    TRAIN_EXIT=$?

    stop_gazebo
    echo $((block + 1)) > "$STATE_FILE"
    echo "  ✅ Blocco $((block + 1))/$TOTAL_BLOCKS completato."
    sleep 2
done

rm -f "$STATE_FILE"
echo "✅ Curriculum completato!"

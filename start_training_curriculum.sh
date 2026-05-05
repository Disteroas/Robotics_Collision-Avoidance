#!/bin/bash
# =============================================================================
#  start_training_curriculum.sh
#
#  Curriculum DDQN multi-labirinto: avvia Gazebo headless, allena per
#  EPISODES_PER_BLOCK episodi, switcha maze, ripete fino a TOTAL_EPISODES.
#
#  Uso:
#    ./start_training_curriculum.sh          # avvia o riprende
#    ./start_training_curriculum.sh --reset  # ricomincia da zero
# =============================================================================

# ─────────────────────────────────────────────────────────────────
#  CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────
TOTAL_EPISODES=3000
EPISODES_PER_BLOCK=100

# FIX: 3x invece di 5x.
# A 5x il wall-clock per step era ~20ms, sotto la latenza Docker→ROS2.
# A 3x → ~33ms per step: garantisce almeno 1 scan LIDAR fresco per step.
GAZEBO_SPEED=4

# FIX: 30s invece di 25s, per dare più margine a Gazebo a 3x.
GAZEBO_WAIT=30

# Percorsi container
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CTR="${SCRIPTS_CTR}/checkpoint.pkl"
PATCHED_WORLD="/tmp/world_fast.world"

# Percorsi host
SCRIPTS_HOST="$(pwd)/src/my_usv/scripts"
STATE_FILE="${SCRIPTS_HOST}/curriculum_state.txt"
PHASE_FILE="${SCRIPTS_HOST}/phase.txt"
PHASE2_PROB=70   # % probabilità maze 2 in Phase 2 (complementare = 30% maze 1)

# ─────────────────────────────────────────────────────────────────
#  MAPPE LABIRINTI
# ─────────────────────────────────────────────────────────────────
declare -A WORLD_PATH SPAWN MAZE_LABEL

WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATH[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"

SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"

MAZE_LABEL[1]="Labirinto 1 (9a)  [TRAIN]"
MAZE_LABEL[2]="Labirinto 2 (9b)  [TRAIN]"
MAZE_LABEL[3]="Labirinto 3 (10)  [TEST]"

# ─────────────────────────────────────────────────────────────────
#  CALCOLI
# ─────────────────────────────────────────────────────────────────
TOTAL_BLOCKS=$(( (TOTAL_EPISODES + EPISODES_PER_BLOCK - 1) / EPISODES_PER_BLOCK ))

mkdir -p "$(pwd)/logs"
mkdir -p "$SCRIPTS_HOST"

# ─────────────────────────────────────────────────────────────────
#  FLAG --reset
# ─────────────────────────────────────────────────────────────────
if [ "$1" = "--reset" ]; then
    echo ""
    echo "⚠️  --reset: cancello checkpoint e stato curriculum."
    read -r -p "   Sei sicuro? (y/N) " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        rm -f "${SCRIPTS_HOST}/checkpoint.pkl" \
              "${SCRIPTS_HOST}/checkpoint.pkl.tmp" \
              "${SCRIPTS_HOST}/phase.txt" \
              "${STATE_FILE}"
        echo "   ✅ Reset effettuato."
    else
        echo "   ❌ Annullato."
        exit 0
    fi
    echo ""
fi

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       USV DDQN  –  CURRICULUM MULTI-LABIRINTO                ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Episodi totali     : %-40s║\n" "$TOTAL_EPISODES"
printf "║  Episodi per blocco : %-40s║\n" "$EPISODES_PER_BLOCK"
printf "║  Blocchi totali     : %-40s║\n" "$TOTAL_BLOCKS"
printf "║  Labirinto TEST     : %-40s║\n" "Maze 3 (mai visto)"
printf "║  Velocità Gazebo    : %-40s║\n" "${GAZEBO_SPEED}x headless"
printf "║  Curriculum         : %-40s║\n" "Phase1=maze1 | Phase2=30/70 (thr:avg50>1500)"
printf "║  MAX_STEPS          : %-40s║\n" "1000"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Log    : src/my_usv/scripts/training_log.csv               ║"
echo "║  Modello: src/my_usv/scripts/best_ddqn_model.pth            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────
#  FUNZIONI
# ─────────────────────────────────────────────────────────────────

stop_container() {
    if docker inspect usv_container &>/dev/null; then
        echo "  [Docker] Fermo container..."
        docker stop usv_container &>/dev/null || true
        for _ in $(seq 1 15); do
            docker inspect usv_container &>/dev/null || break
            sleep 1
        done
        echo "  [Docker] ✅ Container fermato."
    fi
}

select_maze() {
    local phase=1
    if [ -f "$PHASE_FILE" ]; then
        phase=$(cat "$PHASE_FILE" | tr -d '[:space:]')
    fi

    if [ "$phase" = "2" ]; then
        # Phase 2: 30% maze 1, 70% maze 2
        local roll=$(( RANDOM % 100 ))
        if [ "$roll" -lt "$PHASE2_PROB" ]; then
            echo 2
        else
            echo 1
        fi
    else
        # Phase 1: sempre maze 1
        echo 1
    fi
}

start_gazebo_block() {
    local maze_id=$1
    local block=$2
    local log="$(pwd)/logs/block_${block}_maze_${maze_id}.log"

    echo "  [Gazebo] Avvio: ${MAZE_LABEL[$maze_id]} (${GAZEBO_SPEED}x)"

    docker run -d --rm --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/patch_world.py \
                '${WORLD_PATH[$maze_id]}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
            ros2 launch my_usv spawn_robot.launch.py \
                world:=${PATCHED_WORLD} \
                ${SPAWN[$maze_id]} \
                gui:=false
        " >> "$log" 2>&1

    if [ $? -ne 0 ]; then
        echo "  ❌ docker run fallito."
        return 1
    fi

    echo "  [Gazebo] Attendo ${GAZEBO_WAIT}s..."
    sleep "$GAZEBO_WAIT"

    local running
    running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
    if [ "$running" != "true" ]; then
        echo "  ❌ Gazebo crashato. Ultimi log:"
        tail -30 "$log"
        return 1
    fi

    echo "  [Gazebo] ✅ Pronto."
    return 0
}

run_train_block() {
    local start_ep=$1
    local end_ep=$2
    local maze_id=$3

    echo "  [Train]  Ep $((start_ep+1))→${end_ep} su ${MAZE_LABEL[$maze_id]}"

    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/train.py \
                --start-ep   ${start_ep} \
                --end-ep     ${end_ep} \
                --maze-id    ${maze_id} \
                --checkpoint ${CHECKPOINT_CTR} \
                --phase-file ${SCRIPTS_CTR}/phase.txt
        "
    return $?
}

trap 'echo ""; echo "⚠️  Interrotto."; stop_container; exit 0' INT TERM

# ─────────────────────────────────────────────────────────────────
#  RESUME
# ─────────────────────────────────────────────────────────────────
START_BLOCK=0
if [ -f "$STATE_FILE" ]; then
    START_BLOCK=$(cat "$STATE_FILE")
    echo "📂 Resume dal blocco $((START_BLOCK+1))/$TOTAL_BLOCKS"
    echo "   (--reset per ricominciare da zero)"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────
#  LOOP PRINCIPALE
# ─────────────────────────────────────────────────────────────────
T_START=$(date +%s)

for (( block=START_BLOCK; block<TOTAL_BLOCKS; block++ )); do

    EP_START=$(( block * EPISODES_PER_BLOCK ))
    EP_END=$(( EP_START + EPISODES_PER_BLOCK ))
    [ $EP_END -gt $TOTAL_EPISODES ] && EP_END=$TOTAL_EPISODES

    MAZE_ID=$(select_maze)

    # ETA
    ELAPSED=$(( $(date +%s) - T_START ))
    if [ $block -gt $START_BLOCK ] && [ $ELAPSED -gt 0 ]; then
        DONE=$(( block - START_BLOCK ))
        LEFT=$(( TOTAL_BLOCKS - block ))
        ETA_S=$(( ELAPSED / DONE * LEFT ))
        ETA_STR=$(printf '%dh %02dm' $((ETA_S/3600)) $((ETA_S%3600/60)))
    else
        ETA_STR="calcolo..."
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  BLOCCO $((block+1))/$TOTAL_BLOCKS  │  ${MAZE_LABEL[$MAZE_ID]}"
    echo "  Episodi globali: $((EP_START+1))→${EP_END}  │  ETA: ${ETA_STR}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    stop_container
    sleep 2

    if ! start_gazebo_block "$MAZE_ID" "$block"; then
        echo "  ⚠️  Gazebo non partito. Riprova al prossimo avvio."
        sleep 5
        continue
    fi

    run_train_block "$EP_START" "$EP_END" "$MAZE_ID"
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo "  ⚠️  train.py exit $TRAIN_EXIT. Checkpoint salvato."
    fi

    stop_container
    sleep 2

    echo $((block + 1)) > "$STATE_FILE"
    echo "  ✅ Blocco $((block+1))/$TOTAL_BLOCKS completato."
done

# ─────────────────────────────────────────────────────────────────
#  FINE
# ─────────────────────────────────────────────────────────────────
ELAPSED_TOT=$(( $(date +%s) - T_START ))
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🏆  CURRICULUM COMPLETATO                       ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Episodi totali : %-42s║\n" "$TOTAL_EPISODES"
printf "║  Tempo totale   : %-42s║\n" "$(printf '%dh %02dm' $((ELAPSED_TOT/3600)) $((ELAPSED_TOT%3600/60)))"
printf "║  Best model     : %-42s║\n" "src/my_usv/scripts/best_ddqn_model.pth"
printf "║  Prossimo step  : %-42s║\n" "Testa su Labirinto 3 (mai visto)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

rm -f "$STATE_FILE"
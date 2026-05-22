#!/bin/bash
# =============================================================================
#  start_test.sh  –  Valutazione della policy su tutti e 3 i labirinti.
#
#  Sequenza:
#    1. Avvia Gazebo headless su Maze 1  → esegue test.py → ferma Gazebo
#    2. Avvia Gazebo headless su Maze 2  → esegue test.py → ferma Gazebo
#    3. Avvia Gazebo headless su Maze 3  → esegue test.py → ferma Gazebo
#    4. Stampa report comparativo finale dai dati CSV
#
#  Uso:
#    ./start_test.sh
#
#  Output:
#    src/my_usv/scripts/test_results.csv   ← tutti i dati episodio per episodio
#    logs/test_maze_N.log                  ← log Gazebo per ogni maze
# =============================================================================

# ─────────────────────────────────────────────────────────────────
#  CONFIGURAZIONE
# ─────────────────────────────────────────────────────────────────
SEED=0
CONFIG="default"
REPS=30
for arg in "$@"; do
    case "$arg" in
        --seed=*)   SEED="${arg#*=}" ;;
        --config=*) CONFIG="${arg#*=}" ;;
        --reps=*)   REPS="${arg#*=}" ;;
    esac
done

GAZEBO_SPEED=3            # stesso del training per coerenza
GAZEBO_WAIT=25            # secondi per avvio Gazebo
RUN_DIR="runs/${CONFIG}/seed_${SEED}"
mkdir -p "$(pwd)/${RUN_DIR}"

MODEL_PATH="${RUN_DIR}/best_model.pth"
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
MODEL_CTR="/home/usv_ws/${RUN_DIR}/best_model.pth"
OUT_DIR_CTR="/home/usv_ws/${RUN_DIR}"
PATCHED_WORLD="/tmp/world_fast.world"

# run_meta.json: provenienza (git SHA, host, config, seed)
cat > "$(pwd)/${RUN_DIR}/run_meta.json" <<META
{
  "config": "${CONFIG}",
  "seed": ${SEED},
  "reps": ${REPS},
  "git_sha": "$(git rev-parse HEAD 2>/dev/null || echo unknown)",
  "hostname": "$(hostname)",
  "timestamp": "$(date -Iseconds)",
  "success_criterion": "MAX_STEPS=500 reached without collision, epsilon=0.0"
}
META

# ─────────────────────────────────────────────────────────────────
#  MAPPE LABIRINTI
# ─────────────────────────────────────────────────────────────────
declare -A WORLD_PATH SPAWN MAZE_LABEL MAZE_ROLE

WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATH[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"

SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2.5 y:=-0.25 yaw:=0"

MAZE_LABEL[1]="Labirinto 1 (9a)"
MAZE_LABEL[2]="Labirinto 2 (9b)"
MAZE_LABEL[3]="Labirinto 3 (10)"

MAZE_ROLE[1]="TRAIN SET"
MAZE_ROLE[2]="TRAIN SET"
MAZE_ROLE[3]="TEST SET  ← mai visto in training"

mkdir -p "$(pwd)/logs"
mkdir -p "$(pwd)/src/my_usv/scripts"

# ─────────────────────────────────────────────────────────────────
#  VERIFICA MODELLO
# ─────────────────────────────────────────────────────────────────
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "❌  Modello non trovato: $MODEL_PATH"
    echo "    Assicurati che il training sia completato."
    exit 1
fi

# Pulisce il summary precedente (nuova sessione di test)
rm -f "${RUN_DIR}/eval_summary.csv"

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           USV DDQN  –  VALUTAZIONE POLICY                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Modello            : %-40s║\n" "$MODEL_PATH"
printf "║  Reps per maze      : %-40s║\n" "$REPS"
printf "║  Epsilon            : %-40s║\n" "0.0  (policy greedy pura)"
printf "║  Velocità Gazebo    : %-40s║\n" "${GAZEBO_SPEED}x headless"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Maze 1  →  TRAIN SET                                       ║"
echo "║  Maze 2  →  TRAIN SET                                       ║"
echo "║  Maze 3  →  TEST SET  (mai visto in training)               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ─────────────────────────────────────────────────────────────────
#  FUNZIONI
# ─────────────────────────────────────────────────────────────────

stop_container() {
    if docker inspect usv_container &>/dev/null; then
        docker stop usv_container &>/dev/null || true
        for _ in $(seq 1 15); do
            docker inspect usv_container &>/dev/null || break
            sleep 1
        done
    fi
}

start_gazebo() {
    local maze_id=$1
    local log="$(pwd)/logs/test_maze_${maze_id}.log"

    echo "  [Gazebo] Avvio ${MAZE_LABEL[$maze_id]} (${GAZEBO_SPEED}x headless)..."

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
        tail -20 "$log"
        return 1
    fi

    echo "  [Gazebo] ✅ Pronto."
    return 0
}

run_test() {
    local maze_id=$1

    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/test.py \
                --maze-id ${maze_id} \
                --model   ${MODEL_CTR} \
                --reps    ${REPS} \
                --seed    ${SEED} \
                --config  ${CONFIG} \
                --out-dir ${OUT_DIR_CTR}
        "
    return $?
}

trap 'echo ""; echo "⚠️  Interrotto."; stop_container; exit 0' INT TERM

# ─────────────────────────────────────────────────────────────────
#  LOOP SU 3 MAZE
# ─────────────────────────────────────────────────────────────────
for maze_id in 1 2 3; do

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  MAZE ${maze_id}/3  │  ${MAZE_LABEL[$maze_id]}  │  ${MAZE_ROLE[$maze_id]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    stop_container
    sleep 2

    if ! start_gazebo "$maze_id"; then
        echo "  ⚠️  Gazebo non partito per Maze $maze_id. Skipping."
        continue
    fi

    run_test "$maze_id"

    stop_container
    sleep 2
done

# ─────────────────────────────────────────────────────────────────
#  REPORT FINALE
# ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Eval completata. Output in ${RUN_DIR}/"
echo "  eval_summary.csv  eval_steps_m*.csv  eval_crashes_m*.csv"
echo ""
echo "  Per aggregare più seed:"
echo "    python3 src/my_usv/scripts/aggregate_seeds.py \\"
echo "      --config ${CONFIG} --output ANALISI_TRAINING/\$(date +%Y_%m_%d)/aggregate_${CONFIG}.csv"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

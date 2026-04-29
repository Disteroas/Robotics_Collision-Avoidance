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
EPISODES_PER_MAZE=30      # episodi di valutazione per labirinto
GAZEBO_SPEED=3            # stesso del training per coerenza
GAZEBO_WAIT=25            # secondi per avvio Gazebo

MODEL_PATH="src/my_usv/scripts/best_ddqn_model.pth"
OUTPUT_CSV="src/my_usv/scripts/test_results.csv"

# Percorsi container
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
MODEL_CTR="${SCRIPTS_CTR}/best_ddqn_model.pth"
CSV_CTR="${SCRIPTS_CTR}/test_results.csv"
PATCHED_WORLD="/tmp/world_fast.world"

# ─────────────────────────────────────────────────────────────────
#  MAPPE LABIRINTI
# ─────────────────────────────────────────────────────────────────
declare -A WORLD_PATH SPAWN MAZE_LABEL MAZE_ROLE

WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATH[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"

SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"

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

# Pulisce il CSV precedente (nuova sessione di test)
rm -f "$OUTPUT_CSV"

# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           USV DDQN  –  VALUTAZIONE POLICY                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Modello            : %-40s║\n" "$MODEL_PATH"
printf "║  Episodi per maze   : %-40s║\n" "$EPISODES_PER_MAZE"
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
                --maze-id    ${maze_id} \
                --model      ${MODEL_CTR} \
                --episodes   ${EPISODES_PER_MAZE} \
                --output-csv ${CSV_CTR}
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
#  REPORT FINALE COMPARATIVO
# ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 REPORT FINALE COMPARATIVO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Analisi CSV con Python (disponibile nel container, ma usiamo la shell qui)
# Leggiamo il CSV manualmente con awk per non dipendere da Python sull'host
if [ ! -f "$OUTPUT_CSV" ]; then
    echo "  ❌ CSV non trovato: $OUTPUT_CSV"
    exit 1
fi

echo ""
printf "  %-22s %10s %10s %10s %10s\n" "Maze" "Crash%" "Avg Reward" "Avg Steps" "Ruolo"
printf "  %-22s %10s %10s %10s %10s\n" "──────────────────────" "─────────" "──────────" "─────────" "──────────────"

for maze_id in 1 2 3; do

    # Estrai righe del maze corrente (colonna 1 = maze_id)
    # Colonne CSV: maze_id, episode, steps, reward, crashed, min_lidar, avg_lidar
    stats=$(awk -F',' -v mid="$maze_id" '
        NR > 1 && $1 == mid {
            count++
            crashes += $5
            reward_sum += $4
            steps_sum += $3
        }
        END {
            if (count > 0) {
                printf "%.1f %.1f %.1f",
                    crashes/count*100,
                    reward_sum/count,
                    steps_sum/count
            } else {
                print "N/A N/A N/A"
            }
        }
    ' "$OUTPUT_CSV")

    crash_pct=$(echo $stats | awk '{print $1}')
    avg_rew=$(echo $stats | awk '{print $2}')
    avg_stp=$(echo $stats | awk '{print $3}')
    role="${MAZE_ROLE[$maze_id]}"

    printf "  %-22s %9s%% %10s %9s  %s\n" \
        "${MAZE_LABEL[$maze_id]}" "$crash_pct" "$avg_rew" "$avg_stp" "$role"
done

echo ""
echo "  ─────────────────────────────────────────────────────────────"
echo "  💡 Interpretazione:"
echo "     Crash% Maze 3 ≈ Crash% Maze 1/2  →  buona generalizzazione ✅"
echo "     Crash% Maze 3 >> Crash% Maze 1/2 →  overfitting sui maze di train ⚠️"
echo "     Crash% su tutti = 100%            →  training fallito ❌"
echo ""
echo "  📄 Dati completi: $OUTPUT_CSV"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

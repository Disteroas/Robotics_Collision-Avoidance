#!/bin/bash
# =============================================================================
#  start_test.sh  –  Valutazione della policy su tutti e 3 i labirinti.
# =============================================================================

# PORTATO A 90 EPISODI COME RICHIESTO
EPISODES_PER_MAZE=90
GAZEBO_SPEED=3
GAZEBO_WAIT=25

MODEL_PATH="src/my_usv/scripts/best_ddqn_model.pth"
OUTPUT_CSV="src/my_usv/scripts/test_results.csv"

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
MODEL_CTR="${SCRIPTS_CTR}/best_ddqn_model.pth"
CSV_CTR="${SCRIPTS_CTR}/test_results.csv"
PATCHED_WORLD="/tmp/world_fast.world"

declare -A WORLD_PATH SPAWN MAZE_LABEL MAZE_ROLE
WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATH[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"

# Spawn iniziali di Gazebo (sicuri, lontani dai muri)
SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"

MAZE_LABEL[1]="Labirinto 1 (9a)"
MAZE_LABEL[2]="Labirinto 2 (9b)"
MAZE_LABEL[3]="Labirinto 3 (10)"

MAZE_ROLE[1]="TRAIN/VAL SET"
MAZE_ROLE[2]="TRAIN SET"
MAZE_ROLE[3]="TEST SET (Zero-shot)"

mkdir -p "$(pwd)/logs"
mkdir -p "$(pwd)/src/my_usv/scripts"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Modello non trovato: $MODEL_PATH"
    exit 1
fi

rm -f "$OUTPUT_CSV"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           USV DDQN  –  VALUTAZIONE POLICY                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Episodi per maze   : %-40s║\n" "$EPISODES_PER_MAZE"
printf "║  Policy             : %-40s║\n" "Greedy pura (Epsilon 0.0)"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

stop_container() {
    docker rm -f usv_container &>/dev/null
    sleep 2
}

trap 'echo -e "\n⚠️  Interrotto."; stop_container; exit 0' INT TERM

for maze_id in 1 2 3; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  MAZE ${maze_id}/3  │  ${MAZE_LABEL[$maze_id]}  │  ${MAZE_ROLE[$maze_id]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    stop_container
    LOG_FILE="$(pwd)/logs/test_maze_${maze_id}.log"

    docker run -d --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/patch_world.py '${WORLD_PATH[$maze_id]}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
            ros2 launch my_usv spawn_robot.launch.py world:=${PATCHED_WORLD} ${SPAWN[$maze_id]} gui:=false
        " > "$LOG_FILE" 2>&1

    echo "  [Gazebo] Attendo ${GAZEBO_WAIT}s..."
    sleep "$GAZEBO_WAIT"

    running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
    if [ "$running" != "true" ]; then
        echo "  ❌ Errore avvio Gazebo. Skipping Maze $maze_id."
        continue
    fi

    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/test.py \
                --maze-id    ${maze_id} \
                --model      ${MODEL_CTR} \
                --episodes   ${EPISODES_PER_MAZE} \
                --output-csv ${CSV_CTR}
        "
done
stop_container

# Report finale
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 REPORT FINALE COMPARATIVO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "  %-22s %10s %10s %10s %10s\n" "Maze" "Crash%" "Avg Reward" "Avg Steps" "Ruolo"
printf "  %-22s %10s %10s %10s %10s\n" "──────────────────────" "─────────" "──────────" "─────────" "──────────────"

for maze_id in 1 2 3; do
    stats=$(awk -F',' -v mid="$maze_id" '
        NR > 1 && $1 == mid { count++; crashes+=$5; reward_sum+=$4; steps_sum+=$3 }
        END { if (count>0) printf "%.1f %.1f %.1f", crashes/count*100, reward_sum/count, steps_sum/count; else print "N/A N/A N/A" }
    ' "$OUTPUT_CSV")
    
    printf "  %-22s %9s%% %10s %9s  %s\n" "${MAZE_LABEL[$maze_id]}" $(echo $stats | awk '{print $1}') $(echo $stats | awk '{print $2}') $(echo $stats | awk '{print $3}') "${MAZE_ROLE[$maze_id]}"
done
echo ""

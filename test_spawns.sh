#!/bin/bash
# =============================================================================
#  test_spawns.sh — Valida tutti gli spawn point per Maze 1 e Maze 2
#
#  Per ogni spawn: avvia Gazebo headless, spawna il robot,
#  legge primo scan LIDAR, controlla distanza minima da muri.
#
#  Uso:
#    ./test_spawns.sh          # testa entrambi i maze
#    ./test_spawns.sh 1        # solo Maze 1
#    ./test_spawns.sh 2        # solo Maze 2
#
#  Legenda output:
#    ✅  OK        — spawn valido, distanza muro > 0.40m
#    ⚠️   WARNING   — spawn valido ma distanza muro 0.25–0.40m (accettabile)
#    ❌  COLLISION — spawn dentro muro, rimuovere dalla lista
#    ⏱️   TIMEOUT   — Gazebo non ha risposto (problema infrastruttura)
# =============================================================================

MAZE_ARG="${1:-all}"

GAZEBO_WAIT=30        # secondi attesa avvio Gazebo
VALIDATOR_TIMEOUT=20  # secondi attesa primo scan LIDAR

SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"
WORLD_1="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_2="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"

# ─────────────────────────────────────────────────────────────────
#  SPAWN LISTS  (formato: "LABEL  x  y  yaw")
# ─────────────────────────────────────────────────────────────────

# Maze 1 (labirinto_9a): 2 spawn selezionati (canali lat. 1.50m < r_min 1.56m)
# P1: canale sinistro heading N (validato); P2: camera interna heading N (da validare)
SPAWNS_1=(
    "M1-P1  -2.9  -2.0   1.571"
    "M1-P2   1.0  -1.0   1.571"
)

# Maze 2 (labirinto_9b): muri diagonali, estensione x∈[-7.6,+7.3] y∈[-6.3,+6.5]
# Zone A = ingresso sinistro (x ≤ -5.5)
# Zone B = centro-sinistra   (-5.5 < x ≤ -3.5)
# Zone C = centro             (-3.5 < x ≤ -1.0)
# Zone D = centro-destra     (x > -1.0, y tra -2.5 e 2.0)
# Zone E = superiore          (y ≥ 2.5)
# Zone F = inferiore          (y ≤ -3.0)
SPAWNS_2=(
    "M2-A1  -6.0   0.0   0.0"
    "M2-A2  -6.5  -0.5   0.0"
    "M2-B1  -4.5   0.5   0.0"
    "M2-B2  -4.0  -1.0   1.571"
    "M2-B3  -4.5   1.5   2.356"
    "M2-C1  -2.5   1.0   0.0"
    "M2-C2  -7.0   5.0   0.0"
    "M2-C3  -2.0  -1.0   0.785"
    "M2-D1   1.5   0.0   3.142"
    "M2-D2   0.5  -2.0   1.571"
    "M2-D3   3.5   0.5   4.712"
    "M2-E1  -3.0   3.0   0.0"
    "M2-E2   0.0   3.5   3.142"
    "M2-F1  -4.5  -3.5   0.0"
    "M2-F2  -1.5  -4.0   1.571"
    "M2-F3   6.0   6.0   3.142"
)

# ─────────────────────────────────────────────────────────────────
#  FUNZIONI
# ─────────────────────────────────────────────────────────────────

stop_container() {
    docker rm -f usv_container &>/dev/null || true
    sleep 2
}

test_one_spawn() {
    local maze_id=$1 label=$2 x=$3 y=$4 yaw=$5
    local world_path
    [[ $maze_id -eq 1 ]] && world_path="$WORLD_1" || world_path="$WORLD_2"

    # Avvia Gazebo headless con spawn specificato
    docker run -d --rm --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && source install/setup.bash && \
            ros2 launch my_usv spawn_robot.launch.py \
                world:=${world_path} \
                x:=${x} y:=${y} yaw:=${yaw} \
                gui:=false
        " > /dev/null 2>&1

    if [ $? -ne 0 ]; then
        printf "  ❓  %-8s  (%6s, %6s, %5s)  docker run fallito\n" \
               "$label" "$x" "$y" "$yaw"
        return 4
    fi

    sleep "$GAZEBO_WAIT"

    # Verifica Gazebo ancora in piedi
    local running
    running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
    if [ "$running" != "true" ]; then
        printf "  ❓  %-8s  (%6s, %6s, %5s)  Gazebo crashato\n" \
               "$label" "$x" "$y" "$yaw"
        stop_container
        return 4
    fi

    # Esegui validate_spawn.py dentro il container
    local output
    output=$(docker exec usv_container \
        bash -c "
            cd /home/usv_ws && source install/setup.bash && \
            python3 ${SCRIPTS_CTR}/validate_spawn.py \
                --timeout ${VALIDATOR_TIMEOUT}
        " 2>/dev/null)
    local exit_code=$?

    stop_container

    case $exit_code in
        0) printf "  ✅   %-8s  (%6s, %6s, %5s)  %s\n" "$label" "$x" "$y" "$yaw" "$output" ;;
        1) printf "  ❌   %-8s  (%6s, %6s, %5s)  %s\n" "$label" "$x" "$y" "$yaw" "$output" ;;
        2) printf "  ⏱️    %-8s  (%6s, %6s, %5s)  %s\n" "$label" "$x" "$y" "$yaw" "$output" ;;
        3) printf "  ⚠️    %-8s  (%6s, %6s, %5s)  %s\n" "$label" "$x" "$y" "$yaw" "$output" ;;
        *) printf "  ❓   %-8s  (%6s, %6s, %5s)  errore exit=%d\n" "$label" "$x" "$y" "$yaw" "$exit_code" ;;
    esac

    return $exit_code
}

run_maze() {
    local maze_id=$1
    local -n spawn_arr=$2
    local total=${#spawn_arr[@]}
    echo ""
    echo "  ── Maze ${maze_id}  (${total} spawn points) ───────────────────────────"
    for entry in "${spawn_arr[@]}"; do
        read -r label x y yaw <<< "$entry"
        test_one_spawn "$maze_id" "$label" "$x" "$y" "$yaw"
        local rc=$?
        case $rc in
            0) ((COUNT_OK++)) ;;
            1) ((COUNT_FAIL++)); FAILED_SPAWNS+=("$label  x=$x y=$y yaw=$yaw") ;;
            2) ((COUNT_TIMEOUT++)) ;;
            3) ((COUNT_WARN++)) ;;
            *) ((COUNT_TIMEOUT++)) ;;
        esac
    done
}

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

trap 'echo ""; echo "Interrotto."; stop_container; exit 1' INT TERM

COUNT_OK=0; COUNT_FAIL=0; COUNT_WARN=0; COUNT_TIMEOUT=0
FAILED_SPAWNS=()

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              SPAWN POINT VALIDATOR                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
printf "║  Attesa avvio Gazebo  : %-37s║\n" "${GAZEBO_WAIT}s"
printf "║  Timeout scan LIDAR   : %-37s║\n" "${VALIDATOR_TIMEOUT}s"
printf "║  Maze testati         : %-37s║\n" "$MAZE_ARG"
echo "╚══════════════════════════════════════════════════════════════╝"

case "$MAZE_ARG" in
    1)   run_maze 1 SPAWNS_1 ;;
    2)   run_maze 2 SPAWNS_2 ;;
    all) run_maze 1 SPAWNS_1; run_maze 2 SPAWNS_2 ;;
    *)   echo "Uso: $0 [1|2|all]"; exit 1 ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
printf "║  ✅  OK: %-3d   ⚠️   WARN: %-3d   ❌  FAIL: %-3d   ⏱️  TO: %-3d  ║\n" \
       "$COUNT_OK" "$COUNT_WARN" "$COUNT_FAIL" "$COUNT_TIMEOUT"
if [ ${#FAILED_SPAWNS[@]} -gt 0 ]; then
    echo "╠══════════════════════════════════════════════════════════════╣"
    echo "║  Spawn COLLISION — rimuovere dalla lista:                     ║"
    for s in "${FAILED_SPAWNS[@]}"; do
        printf "║    ❌  %-58s║\n" "$s"
    done
fi
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

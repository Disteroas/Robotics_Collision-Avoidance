#!/bin/bash
# =============================================================================
#  start_test_gui.sh  –  Test visuale con GUI Gazebo (singolo maze)
#
#  Apre Gazebo con interfaccia grafica per ispezionare il comportamento
#  del robot dopo il training. Richiede XLaunch (VcXsrv) su Windows.
#
#  Uso: ./start_test_gui.sh [maze_id] [seed] [config] [reps] [speed]
#       maze_id : 1, 2, 3 (default: 1)
#       seed    : seed del modello in runs/<config>/seed_<S>/ (default: 1)
#       config  : nome config (default: r_alpha)
#       reps    : ripetizioni per spawn — round-robin copre TUTTI gli spawn
#                 del maze (M1=2, M2=6). reps=1 → ogni spawn una volta (default: 1)
#       speed   : moltiplicatore real-time Gazebo (default: 1 = real-time, per
#                 guardare comodo). ATTENZIONE: <3x può dare desync LIDAR durante
#                 l'eval — i crash a 1x sono per ISPEZIONE, non metrica rigorosa.
#                 Per metrica usa start_test.sh (3x).
#
#  Prerequisito Windows: XLaunch (VcXsrv) con "Disable access control"
#  NOTA: test.py teleporta lo UGV ad ogni spawn della lista round-robin,
#        quindi vedi anche gli spawn che falliscono (es. M1 P2 (1.0,-1.0)).
#        L'output GUI va in runs/<config>/seed_<S>/gui/ per NON toccare
#        eval_summary.csv reale.
# =============================================================================

MAZE_ID=${1:-1}
SEED=${2:-1}
CONFIG=${3:-r_alpha}
REPS=${4:-1}
GAZEBO_WAIT=35
GAZEBO_SPEED=${5:-1}     # 1x = real-time, per ispezione visiva (come vecchi branch)
PATCHED_WORLD="/tmp/world_gui.world"
MODEL_PATH="runs/${CONFIG}/seed_${SEED}/best_model.pth"
OUT_DIR="runs/${CONFIG}/seed_${SEED}/gui"
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"

declare -A WORLD_PATH SPAWN MAZE_LABEL
WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATH[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"
SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"
MAZE_LABEL[1]="Labirinto 1 (9a)"
MAZE_LABEL[2]="Labirinto 2 (9b)"
MAZE_LABEL[3]="Labirinto 3 (10)"

if [ ! -f "$MODEL_PATH" ]; then
    echo "Errore: modello non trovato: $MODEL_PATH"
    exit 1
fi

echo ""
echo "============================================================"
echo "  UGV DDQN — TEST VISUALE (GUI)"
echo "============================================================"
echo "  Maze    : ${MAZE_ID} — ${MAZE_LABEL[$MAZE_ID]}"
echo "  Seed    : ${SEED}  | Config: ${CONFIG}  | Reps/spawn: ${REPS}"
echo "  Speed   : ${GAZEBO_SPEED}x real-time"
echo "  Modello : ${MODEL_PATH}"
echo "  Output  : ${OUT_DIR}/ (separato da eval reale)"
echo "  NOTA    : Richiede XLaunch (VcXsrv) su Windows"
echo "============================================================"
echo ""

trap 'echo ""; echo "Interruzione: rimozione container..."; docker rm -f usv_container 2>/dev/null; exit 0' INT TERM EXIT

docker rm -f usv_container 2>/dev/null
sleep 1

echo "Avvio Gazebo con GUI..."

docker run -d --name usv_container \
    --env="DISPLAY=host.docker.internal:0.0" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/$(pwd):/home/usv_ws" \
    usv_rl_project \
    bash -c "
        cd /home/usv_ws && source install/setup.bash && \
        python3 ${SCRIPTS_CTR}/patch_world.py \
            '${WORLD_PATH[$MAZE_ID]}' ${GAZEBO_SPEED} ${PATCHED_WORLD} && \
        ros2 launch my_usv spawn_robot.launch.py \
            world:=${PATCHED_WORLD} \
            ${SPAWN[$MAZE_ID]} gui:=true
    "

echo "Attendo ${GAZEBO_WAIT}s avvio Gazebo GUI..."
sleep "$GAZEBO_WAIT"

running=$(docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null)
if [ "$running" != "true" ]; then
    echo "Errore: Gazebo non avviato. Verificare XLaunch."
    exit 1
fi

echo "Gazebo GUI attivo. Avvio test..."
echo ""

docker exec usv_container \
    bash -c "
        cd /home/usv_ws && source install/setup.bash && \
        python3 ${SCRIPTS_CTR}/test.py \
            --maze-id  ${MAZE_ID} \
            --model    /home/usv_ws/${MODEL_PATH} \
            --reps     ${REPS} \
            --seed     ${SEED} \
            --config   ${CONFIG} \
            --out-dir  /home/usv_ws/${OUT_DIR}
    "

echo ""
echo "Test completato. Premere Ctrl+C o chiudere Gazebo."
echo "Risultati: ${OUT_DIR}/eval_summary.csv (+ eval_steps/crashes per maze)"
docker logs -f usv_container

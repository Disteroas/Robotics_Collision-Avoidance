#!/bin/bash
# =============================================================================
#  start_test_gui.sh  –  Test visuale con GUI Gazebo (singolo maze)
#
#  Apre Gazebo con interfaccia grafica per ispezionare il comportamento
#  del robot dopo il training. Richiede XLaunch (VcXsrv) su Windows.
#
#  Uso: ./start_test_gui.sh [maze_id] [episodes]
#       maze_id : 1, 2, 3 (default: 2)
#       episodes: episodi da testare (default: 5)
#
#  Prerequisito Windows: XLaunch (VcXsrv) con "Disable access control"
# =============================================================================

MAZE_ID=${1:-2}
EPISODES=${2:-5}
GAZEBO_WAIT=35
MODEL_PATH="src/my_usv/scripts/best_ddqn_model.pth"
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
echo "  USV DDQN — TEST VISUALE (GUI)"
echo "============================================================"
echo "  Maze    : ${MAZE_ID} — ${MAZE_LABEL[$MAZE_ID]}"
echo "  Episodi : ${EPISODES}"
echo "  Modello : ${MODEL_PATH}"
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
        ros2 launch my_usv spawn_robot.launch.py \
            world:=${WORLD_PATH[$MAZE_ID]} \
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
            --maze-id   ${MAZE_ID} \
            --model     ${SCRIPTS_CTR}/best_ddqn_model.pth \
            --episodes  ${EPISODES} \
            --output-csv ${SCRIPTS_CTR}/test_gui_results.csv
    "

echo ""
echo "Test completato. Premere Ctrl+C o chiudere Gazebo."
echo "Risultati: src/my_usv/scripts/test_gui_results.csv"
docker logs -f usv_container

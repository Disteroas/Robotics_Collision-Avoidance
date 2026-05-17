#!/bin/bash
# =============================================================================
#  start_test_gui.sh  –  Test visuale con GUI Gazebo
# =============================================================================

MAZE_ID=${1:-2}
EPISODES=${2:-5}
GAZEBO_WAIT=35

MODEL_PATH="src/my_usv/scripts/best_ddqn_model.pth"
SCRIPTS_CTR="/home/usv_ws/src/my_usv/scripts"

declare -A WORLD_PATH SPAWN
WORLD_PATH[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATH[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATH[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"
SPAWN[1]="x:=-3 y:=-5 yaw:=1.57"
SPAWN[2]="x:=-6 y:=0 yaw:=0"
SPAWN[3]="x:=-2 y:=-1 yaw:=0"

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ Errore: modello non trovato in $MODEL_PATH"
    exit 1
fi

trap 'docker rm -f usv_container &>/dev/null; exit 0' INT TERM EXIT
docker rm -f usv_container &>/dev/null

echo "Avvio Gazebo GUI (Maze ${MAZE_ID})..."
docker run -d --name usv_container \
    --env="DISPLAY=host.docker.internal:0.0" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/$(pwd):/home/usv_ws" \
    usv_rl_project \
    bash -c "
        cd /home/usv_ws && source install/setup.bash && \
        ros2 launch my_usv spawn_robot.launch.py world:=${WORLD_PATH[$MAZE_ID]} ${SPAWN[$MAZE_ID]} gui:=true
    " > /dev/null 2>&1

sleep "$GAZEBO_WAIT"

echo "Gazebo GUI attivo. Esecuzione policy..."
docker exec usv_container \
    bash -c "
        cd /home/usv_ws && source install/setup.bash && \
        python3 ${SCRIPTS_CTR}/test.py --maze-id ${MAZE_ID} --model ${SCRIPTS_CTR}/best_ddqn_model.pth --episodes ${EPISODES} --output-csv ${SCRIPTS_CTR}/test_gui_results.csv
    "

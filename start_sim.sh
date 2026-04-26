#!/bin/bash
# Script intelligente per avviare Gazebo scegliendo il labirinto

# Legge il primo numero che scrivi dopo il comando. Se non scrivi nulla, usa "1" di default.
SCELTA=${1:-1}

case $SCELTA in
    1)
        echo "Avvio Labirinto 1..."
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
        COORDS="x:=-3 y:=-5 yaw:=1.57"
        ;;
    2)
        echo "Avvio Labirinto 2..."
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
        COORDS="x:=-6 y:=0 yaw:=0"
        ;;
    3)
        echo "Avvio Labirinto 3..."
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"
        COORDS="x:=-2 y:=-1 yaw:=0"
        ;;
    *)
        echo "Errore: Labirinto non trovato. Scegli 1, 2 o 3."
        exit 1
        ;;
esac

# ==========================================
# Riconoscimento automatico Sistema Operativo
# ==========================================
if [ "$(uname)" == "Linux" ]; then
    echo "OS rilevato: Linux (Ubuntu) - Imposto la grafica nativa"
    GUI_SETTINGS="--env=DISPLAY=$DISPLAY --volume=/tmp/.X11-unix:/tmp/.X11-unix:rw --device=/dev/dri"
    DIR_PATH="$(pwd)"
else
    echo "OS rilevato: Windows/Mac - Imposto la grafica host.docker.internal"
    GUI_SETTINGS="--env=DISPLAY=host.docker.internal:0.0 --env=QT_X11_NO_MITSHM=1"
    DIR_PATH="/$(pwd)"
fi

# Lancia Docker passandogli le variabili scelte
docker run -it --rm --name usv_container \
  $GUI_SETTINGS \
  --volume="${DIR_PATH}:/home/usv_ws" \
  usv_rl_project \
  bash -c "source install/setup.bash && ros2 launch my_usv spawn_robot.launch.py world:=$WORLD_PATH $COORDS"

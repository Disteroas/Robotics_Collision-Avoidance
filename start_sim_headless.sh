#!/bin/bash
# Script OTTIMIZZATO (Headless) per l'addestramento. Nessuna GUI, massima velocità.

SCELTA=${1:-1}

case $SCELTA in
    1)
        echo "Avvio Labirinto 1 in background (Headless)..."
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
        COORDS="x:=-3 y:=-5 yaw:=1.57"
        ;;
    2)
        echo "Avvio Labirinto 2 in background (Headless)..."
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
        COORDS="x:=-6 y:=0 yaw:=0"
        ;;
    3)
        echo "Avvio Labirinto 3 in background (Headless)..."
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"
        COORDS="x:=-2 y:=-1 yaw:=0"
        ;;
    *)
        echo "Errore: Labirinto non trovato. Scegli 1, 2 o 3."
        exit 1
        ;;
esac

# Lancio Docker senza le variabili video (DISPLAY/X11) per risparmiare RAM
docker run -it --rm --name usv_container_headless \
  --volume="/$(pwd):/home/usv_ws" \
  usv_rl_project \
  bash -c "source install/setup.bash && ros2 launch my_usv spawn_robot.launch.py world:=$WORLD_PATH $COORDS gui:=false"
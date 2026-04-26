#!/bin/bash
# =============================================================================
#  start_sim.sh  –  Avvio HEADLESS di Gazebo con velocità di simulazione
#                   configurabile.
#
#  Uso:
#    ./start_sim.sh [labirinto] [speed]
#
#  Argomenti:
#    labirinto  –  1, 2 o 3   (default: 1)
#    speed      –  fattore di velocità rispetto al real-time (default: 5)
#
#  Esempi:
#    ./start_sim.sh          → labirinto 1, velocità 5x
#    ./start_sim.sh 2        → labirinto 2, velocità 5x
#    ./start_sim.sh 3 8      → labirinto 3, velocità 8x
#
#  Nota su speed:
#    5  → sicuro per qualsiasi mondo, raccomandato per iniziare
#    8  → molto veloce, verificare che la fisica rimanga stabile
#    10 → massimo pratico, possibile instabilità ODE in ambienti complessi
# =============================================================================

SCELTA=${1:-1}
SPEED=${2:-5}

# --------------------------------------------------------------------------- #
# Selezione del mondo                                                           #
# --------------------------------------------------------------------------- #
case $SCELTA in
    1)
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
        COORDS="x:=-3 y:=-5 yaw:=1.57"
        LABEL="Labirinto 1 (9a)"
        ;;
    2)
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
        COORDS="x:=-6 y:=0 yaw:=0"
        LABEL="Labirinto 2 (9b)"
        ;;
    3)
        WORLD_PATH="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"
        COORDS="x:=-2 y:=-1 yaw:=0"
        LABEL="Labirinto 3 (10)"
        ;;
    *)
        echo "❌  Errore: scegli 1, 2 o 3 come labirinto."
        exit 1
        ;;
esac

# --------------------------------------------------------------------------- #
# Percorso dello script di patching (dentro il volume montato)                 #
# --------------------------------------------------------------------------- #
PATCHER_HOST="$(pwd)/src/my_usv/scripts/patch_world.py"
PATCHER_CONTAINER="/home/usv_ws/src/my_usv/scripts/patch_world.py"
PATCHED_WORLD="/tmp/world_fast.world"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           USV SIMULATION  –  MODALITÀ HEADLESS           ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Mondo   : $LABEL"
echo "║  Velocità: ${SPEED}x real-time  (real_time_factor=$SPEED)"
echo "║  GUI     : DISABILITATA  (nessuna finestra grafica)"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------- #
# Avvio del container Docker                                                    #
#                                                                               #
# Differenze rispetto alla versione originale:                                  #
#   1. RIMOSSO  --env="DISPLAY=..."  → nessun forwarding X11                  #
#   2. RIMOSSO  --env="QT_X11_NO_MITSHM=1"  → non serve senza GUI             #
#   3. AGGIUNTO patching del world file prima del launch                        #
#   4. AGGIUNTO gui:=false al comando di launch                                 #
#                                                                               #
# Se il tuo spawn_robot.launch.py NON supporta gui:=false:                     #
#   Apri il launch file e aggiungi:                                             #
#     gui = LaunchConfiguration('gui', default='false')                        #
#   e passalo a gazebo_ros nel launch argument 'gui'.                           #
# --------------------------------------------------------------------------- #
docker run -it --rm --name usv_container \
  --volume="$(pwd):/home/usv_ws" \
  usv_rl_project \
  bash -c "
    set -e

    source install/setup.bash

    echo '>>> Patchando physics nel world file...'
    python3 ${PATCHER_CONTAINER} '${WORLD_PATH}' ${SPEED} ${PATCHED_WORLD}

    echo '>>> Avvio Gazebo (headless, ${SPEED}x)...'
    ros2 launch my_usv spawn_robot.launch.py \
      world:=${PATCHED_WORLD} \
      ${COORDS} \
      gui:=false
  "

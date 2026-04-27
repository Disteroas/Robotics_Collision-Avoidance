#!/bin/bash
# =============================================================================
#  start_sim.sh  –  Avvio HEADLESS di Gazebo con velocità configurabile.
#
#  Uso:
#    ./start_sim.sh [labirinto] [speed]
#
#  Esempi:
#    ./start_sim.sh          →  labirinto 1, velocità 5x (default)
#    ./start_sim.sh 2        →  labirinto 2, velocità 5x
#    ./start_sim.sh 3 8      →  labirinto 3, velocità 8x
#
#  Note su speed:
#    5  → sicuro e stabile, consigliato per iniziare
#    8  → veloce, verificare che la fisica non esploda
#    10 → massimo pratico, possibile instabilità ODE
# =============================================================================

SCELTA=${1:-1}
SPEED=${2:-5}

# --------------------------------------------------------------------------- #
#  Selezione del mondo                                                          #
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
#  Percorso dello script di patching (nel volume montato, lato container)      #
# --------------------------------------------------------------------------- #
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
#  Avvio Docker                                                                 #
#                                                                               #
#  FIX 1 – "/$(pwd)" con la barra iniziale:                                   #
#    Su Git Bash per Windows, $(pwd) restituisce un path Unix-style            #
#    (/c/Users/david/...). Docker Desktop su Windows vuole il path in          #
#    formato Windows (C:/Users/david/...). La barra iniziale "/" forza         #
#    Git Bash a NON convertire il path, lasciandolo passare così com'è         #
#    a Docker che sa già come interpretarlo. Senza "/", Docker riceve          #
#    un path malformato e il volume mount fallisce silenziosamente.            #
#                                                                               #
#  FIX 2 – "cd /home/usv_ws &&" come prima istruzione bash -c:               #
#    Il container si avvia nella sua WORKDIR di default (tipicamente "/").     #
#    "source install/setup.bash" cercava il file in "/" invece che in          #
#    "/home/usv_ws/" → No such file or directory → crash immediato →          #
#    container usciva → start_train.sh non trovava il container.               #
#                                                                               #
#  FIX 3 – Rimossi DISPLAY e QT_X11_NO_MITSHM:                               #
#    Non servono in modalità headless. Rimuoverli evita warning su             #
#    sistemi dove il display X11 non è configurato.                            #
# --------------------------------------------------------------------------- #
docker run -it --rm --name usv_container \
  --volume="/$(pwd):/home/usv_ws" \
  usv_rl_project \
  bash -c "
    cd /home/usv_ws && \
    source install/setup.bash && \
    echo '>>> Patching world file per velocità ${SPEED}x...' && \
    python3 ${PATCHER_CONTAINER} '${WORLD_PATH}' ${SPEED} ${PATCHED_WORLD} && \
    echo '>>> Avvio Gazebo headless (${SPEED}x real-time)...' && \
    ros2 launch my_usv spawn_robot.launch.py \
      world:=${PATCHED_WORLD} \
      ${COORDS} \
      gui:=false
  "

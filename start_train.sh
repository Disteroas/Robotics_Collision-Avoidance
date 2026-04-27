#!/bin/bash
# =============================================================================
#  start_train.sh  –  Avvio del training DDQN nel container già attivo.
#
#  Uso:
#    ./start_train.sh
#
#  Pre-requisito:
#    Il container `usv_container` deve essere in esecuzione
#    (avviato con start_sim.sh).
#
#  Output live sul terminale:
#    Ep  123/3000 | ✅ OK    | Reward:  2341.0 | AvgR(100):  1823.4 | ε: 0.742 | ...
#    Ep  124/3000 | 💥 CRASH | Reward:  -234.0 | AvgR(100):  1756.1 | ε: 0.741 | ...
#
#  File salvati (dentro il volume, visibili anche su Windows):
#    src/my_usv/scripts/training_log.csv      ← log completo ogni episodio
#    src/my_usv/scripts/best_ddqn_model.pth   ← miglior modello (media 100 ep)
# =============================================================================

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              USV DDQN  –  AVVIO TRAINING                 ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Log CSV   : src/my_usv/scripts/training_log.csv         ║"
echo "║  Modello   : src/my_usv/scripts/best_ddqn_model.pth      ║"
echo "║  Stop      : Ctrl+C                                       ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------- #
#  FIX: aggiunto "cd /home/usv_ws &&" prima di "source install/setup.bash"   #
#  Stesso problema di start_sim.sh: docker exec apre una shell nella          #
#  WORKDIR del container (non necessariamente /home/usv_ws), quindi           #
#  "source install/setup.bash" senza cd cercava il file nella directory       #
#  sbagliata e falliva.                                                        #
# --------------------------------------------------------------------------- #
docker exec -it usv_container \
  bash -c "
    cd /home/usv_ws && \
    source install/setup.bash && \
    python3 src/my_usv/scripts/train.py
  "

#!/bin/bash
# =============================================================================
#  start_train.sh  –  Avvio del training DDQN nel container attivo.
#
#  Uso:
#    ./start_train.sh
#
#  Pre-requisito:
#    Il container `usv_container` deve essere già in esecuzione
#    (avviato con start_sim.sh).
#
#  Output:
#    - Stampa real-time sul terminale (episodio, reward, ε, loss, crash)
#    - Salva log CSV in: src/my_usv/scripts/training_log.csv
#    - Salva il miglior modello in: src/my_usv/scripts/best_ddqn_model.pth
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

# Entra nel container e lancia il training.
# Il nodo legge il clock simulato di Gazebo (use_sim_time impostato
# internamente in UsvEnv.__init__), quindi funziona automaticamente
# a qualsiasi velocità impostata in start_sim.sh.
docker exec -it usv_container \
  bash -c "source install/setup.bash && python3 src/my_usv/scripts/train.py"

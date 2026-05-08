#!/bin/bash
# Testa la policy sul labirinto già aperto in Gazebo.
# Prerequisito: ./start_sim.sh <maze_id> deve essere in esecuzione in un altro terminale.

CONTAINER="usv_container"

running=$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)
if [ "$running" != "true" ]; then
    echo "Errore: container '$CONTAINER' non trovato o non in esecuzione."
    echo "Avvia prima Gazebo con ./start_sim.sh <maze_id>"
    exit 1
fi

echo "Container in esecuzione. Avvio test.py..."

# winpty necessario su Git Bash Windows per allocare TTY
if command -v winpty &>/dev/null; then
    winpty docker exec -it "$CONTAINER" \
        bash -c "source install/setup.bash && python3 src/my_usv/scripts/test.py"
else
    docker exec -i "$CONTAINER" \
        bash -c "source install/setup.bash && python3 src/my_usv/scripts/test.py"
fi

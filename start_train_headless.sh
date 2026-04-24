#!/bin/bash
# Questo script entra nel container Headless e avvia il motore neurale
docker exec -it usv_container_headless bash -c "source install/setup.bash && python3 src/my_usv/scripts/train.py"
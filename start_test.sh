#!/bin/bash
# Questo script entra nel container già attivo e lancia il training
docker exec -it usv_container bash -c "source install/setup.bash && python3 src/my_usv/scripts/test.py"
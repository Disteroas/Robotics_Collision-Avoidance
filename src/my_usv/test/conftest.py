import sys
import os

# Aggiunge src/my_usv/scripts/ al path così i test trovano i moduli senza ROS install
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

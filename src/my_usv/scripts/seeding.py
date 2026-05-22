"""Controllo centralizzato del seed. Nessuna dipendenza ROS.

NOTA: Gazebo (fisica + timing ROS) resta non deterministico. Il seed NON dà
riproducibilità bit-a-bit, ma rende la varianza attribuibile e misurabile
(Henderson 2018). torch.use_deterministic_algorithms NON è attivato: su CPU
dà overhead e la non-determinismo dominante qui è Gazebo, non i kernel torch.
"""
import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

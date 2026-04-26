#!/usr/bin/env python3
"""
patch_world.py  –  Inietta le impostazioni di fisica in un world file Gazebo.

Uso:
    python3 patch_world.py <world_input> <real_time_factor> <world_output>

Esempio:
    python3 patch_world.py labirinto_9a.world 5 /tmp/world_fast.world

Il blocco <physics> viene:
  - SOSTITUITO se già presente (qualunque variante di attributi)
  - INSERITO prima di </world> se assente

Parametri di fisica scelti:
  - max_step_size      = 0.001 s   (1 ms per step, standard ODE)
  - real_time_update_rate = 1000   (1000 step/s in real time = 1x)
  - real_time_factor   = N         (N > 1 → simulazione più veloce del real-time)

Con real_time_factor=5 Gazebo simula 5 secondi di fisica ogni secondo reale.
ATTENZIONE: valori > 10 possono causare instabilità numerica nell'ODE solver.
"""

import re
import sys
import os


def build_physics_block(real_time_factor: float) -> str:
    """Restituisce il blocco XML <physics> con i parametri ottimizzati."""
    return (
        "<physics type='ode'>\n"
        "      <max_step_size>0.001</max_step_size>\n"
        "      <real_time_update_rate>1000</real_time_update_rate>\n"
        f"      <real_time_factor>{real_time_factor}</real_time_factor>\n"
        "    </physics>"
    )


def patch_world(input_path: str, real_time_factor: float, output_path: str) -> None:
    if not os.path.isfile(input_path):
        print(f"[ERRORE] File non trovato: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    new_block = build_physics_block(real_time_factor)

    # Regex che matcha qualsiasi <physics ...>...</physics> su più righe
    physics_pattern = re.compile(r"<physics[^>]*>.*?</physics>", re.DOTALL)

    if physics_pattern.search(content):
        # Sostituisce il blocco esistente
        patched = physics_pattern.sub(new_block, content)
        action = "sostituito"
    else:
        # Inserisce prima del tag di chiusura </world>
        if "</world>" not in content:
            print("[ERRORE] Tag </world> non trovato nel file.", file=sys.stderr)
            sys.exit(1)
        patched = content.replace("</world>", f"  {new_block}\n</world>")
        action = "inserito"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(
        f"[OK] Physics {action} → real_time_factor={real_time_factor} | "
        f"Output: {output_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Uso: python3 patch_world.py <input.world> <real_time_factor> <output.world>"
        )
        sys.exit(1)

    input_path  = sys.argv[1]
    speed       = float(sys.argv[2])
    output_path = sys.argv[3]

    patch_world(input_path, speed, output_path)

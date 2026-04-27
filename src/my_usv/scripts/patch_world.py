#!/usr/bin/env python3
import re
import sys
import os

def patch_world(input_path: str, speed: float, output_path: str) -> None:
    if not os.path.isfile(input_path):
        print(f"[ERRORE] File non trovato: {input_path}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        content = f.read()

    # In Gazebo, la velocità target è determinata da real_time_update_rate.
    # 1x real-time = 1000 step/sec. 5x = 5000 step/sec.
    # Se impostato a 0, Gazebo disabilita il freno e va al massimo frame-rate possibile della tua CPU.
    target_update_rate = int(speed * 1000)

    # Regex mirata: sostituisce SOLO l'update_rate lasciando intatto il blocco <ode>
    if "<real_time_update_rate>" in content:
        patched = re.sub(
            r"<real_time_update_rate>.*?</real_time_update_rate>",
            f"<real_time_update_rate>{target_update_rate}</real_time_update_rate>",
            content
        )
    else:
        # Se non c'è, lo inietta subito dentro il tag di apertura <physics>
        patched = re.sub(
            r"(<physics[^>]*>)",
            f"\\1\n      <real_time_update_rate>{target_update_rate}</real_time_update_rate>",
            content
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(patched)

    print(f"[OK] Physics aggiornata (ODE intatto) → update_rate={target_update_rate}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Uso: python3 patch_world.py <input.world> <speed> <output.world>")
        sys.exit(1)

    input_path  = sys.argv[1]
    speed       = float(sys.argv[2])
    output_path = sys.argv[3]
    
    patch_world(input_path, speed, output_path)
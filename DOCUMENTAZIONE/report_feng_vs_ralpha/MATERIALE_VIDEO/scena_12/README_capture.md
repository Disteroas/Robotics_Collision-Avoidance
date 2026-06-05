# Scena 12 — Gazebo footage da catturare (placeholder windows)

I render `anim_hw_collapse` (beat D) e `anim_crash_modes` (beat E) hanno finestre
vuote: il footage Gazebo si monta dopo in DaVinci.

## Beat E — crash modes (UN solo comando per entrambe le clip)

Round-robin su M2 di feng seed_0 copre tutti gli spawn, inclusi i due che servono:
- **Perceptual** = spawn `(-6.0, 0.0)`  (sterza nel muro, spazio libero accanto)
- **Kinematic**  = spawn `(-7.0, 5.0)`  (pocket R_min=0.625m, non riesce a girare)

```bash
# Git Bash, root progetto, XLaunch (VcXsrv) attivo con "Disable access control"
./start_test_gui.sh 2 0 feng_hw_A 1 1
#                    │ │ │         │ └ speed 1x (real-time, per registrare)
#                    │ │ │         └ reps=1 (ogni spawn una volta)
#                    │ │ └ config = feng_hw_A
#                    │ └ seed = 0
#                    └ maze = 2
```
Registra la GUI Gazebo con OBS. Gli episodi escono in ordine round-robin
deterministico → taglia il segmento dello spawn `(-6,0)` (perceptual) e quello
`(-7,5)` (kinematic). Le strisce azioni nel render usano gli stessi episodi
(155 step perceptual, 127 step kinematic).

## Beat D — cross-machine (opzionale)

Due finestre Machine A / Machine B con M3 59% vs 0%. Footage = due terminali o due
GUI eval su M3, stesso modello, due PC. Non riproducibile su una macchina sola →
usa b-roll/terminali esistenti, o lascia i placeholder.

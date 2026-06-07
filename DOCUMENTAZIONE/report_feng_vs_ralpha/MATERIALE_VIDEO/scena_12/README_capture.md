# Scena 12 — Gazebo footage da catturare (placeholder windows)

Solo `anim_crash_modes` (beat E) ha finestre placeholder da riempire con footage
Gazebo in DaVinci. `anim_hw_collapse` (beat D) è ora una barra animata
auto-contenuta — niente footage richiesto.

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

## Inquadratura consigliata — TOP-DOWN (ortografica)

Entrambe le cause sono SPAZIALI e si leggono solo dall'alto:

- **Perceptual** (spawn `(-6,0)`, 155 step): la barca sterza nel muro mentre
  accanto c'è spazio libero. Vista dall'alto → si vede il varco ignorato.
  Fine clip sull'impatto. In DaVinci: freccia che punta sullo spazio libero
  ("it was the seeing").
- **Kinematic** (spawn `(-7,5)`, 127 step): entra nel pocket R_min, non riesce
  a girarsi, sbatte. Vista dall'alto → si vede che NON può completare la curva.

In Gazebo: vista ortografica dall'alto (top view). Sincronizza la velocità di
ciascuna clip alla lunghezza della sua striscia azioni (155 / 127 step).

## Beat D — cross-machine (niente footage)

`anim_hw_collapse` è una singola barra che viaggia hw_A→hw_B collassando 59%→0%:
il messaggio è auto-contenuto nel render, nessun footage Gazebo richiesto.

# Spawn Diversity & Safety — Design Spec

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan.

**Goal:** Replace 8 clustered spawn points on Maze 2 with 16 well-distributed points covering 6 zones, and add post-teleport safety validation to eliminate -1000 crashes at step=1.

**Branch:** `feng_direct`

**File to modify:** `src/my_usv/scripts/usv_env.py`

---

## 1. Root cause analysis — step=1 crashes (-1000 diretti)

### Data

| Block | Maze 2 episodes | Step=1 crash |
|-------|----------------|-------------|
| Block 0 (no teleport, spawn fisso x=-6,y=0) | 1900 | **3 (0.2%)** |
| Block 2 (teleport random, 8 spawn points) | 1400 | **349 (24.9%)** |

Il teleport è la variabile causale. Spawn fisso nell'area aperta → quasi zero crashes.

### Meccanismo

`reset_environment` esegue il teleport ma **non verifica** se la posizione di spawn è sicura prima di restituire l'osservazione. La collision check avviene solo in `step_action` (alla prima azione):

```
reset_environment():
    _teleport(x, y, yaw)        # robot spawna a posizione
    get_state()                  # ← nessun controllo min_lidar
    return obs                   # osservazione restituita

step_action(action):
    move 0.1 sim-sec             # robot si muove 0.05m
    compute_reward(scan)         # ← PRIMO controllo collision
    → se min_lidar < 0.25m: return -1000, done=True
```

Se dopo il teleport il robot si trova a 0.25–0.30m da un muro (imprecisione fisica di Gazebo: il collision mesh del robot si posiziona 2–3cm diversamente dalle coordinate nominali), qualsiasi azione con `linear.x = 0.5 m/s` avanza 0.05m → min_lidar scende sotto 0.25m → crash a step=1.

### Perché 24.9% e non 12.5% (solo M2-B2)?

Gazebo posiziona il robot con imprecisione di ±2–3cm rispetto alle coordinate nominali. Il validation script `test_spawns.sh` usa min_lidar statico, ma durante il training il robot ha velocità residua da teleport + imprecisione fisica → posizione effettiva diversa da quella testata staticamente. Questo impatta più di un punto spawn, distribuendosi uniformemente sugli 8 punti (confermato da analisi modulo-8 sul CSV).

### Perché Block 0 non aveva il problema?

Block 0 usava spawn fisso (`x=-6, y=0`), un'area aperta senza muri nelle immediate vicinanze. Nessun rischio di posizionamento marginale.

---

## 2. Design

### Componente 1: Post-teleport spawn safety check (`reset_environment`)

Dopo il teleport, prima di restituire l'osservazione, leggere il primo scan LIDAR reale. Se `min_lidar < SPAWN_SAFETY_DIST = 0.40m` (margine sopra il collision threshold 0.25m), scegliere un altro spawn e riprovare (max 3 tentativi).

**Costante da aggiungere in cima al file:**
```python
SPAWN_SAFETY_DIST = 0.40  # min LIDAR accettabile dopo teleport (m)
SPAWN_MAX_RETRIES = 3
```

**Modifica a `reset_environment`:**

Attuale:
```python
x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
self._teleport(x, y, yaw)

for _ in range(20):
    rclpy.spin_once(self, timeout_sec=0.0)

self._wait_sim_seconds(0.8)
self.accepting_scans = True

for _ in range(5):
    rclpy.spin_once(self, timeout_sec=0.1)

return self.get_state()
```

Nuovo:
```python
for attempt in range(SPAWN_MAX_RETRIES):
    x, y, yaw = random.choice(SPAWN_LISTS[maze_id])
    self._teleport(x, y, yaw)

    for _ in range(20):
        rclpy.spin_once(self, timeout_sec=0.0)

    self._wait_sim_seconds(0.8)
    self.accepting_scans = True

    for _ in range(5):
        rclpy.spin_once(self, timeout_sec=0.1)

    min_dist = float(self.current_scan.min()) * LIDAR_MAX_RANGE
    if min_dist >= SPAWN_SAFETY_DIST:
        break

    self.get_logger().warn(
        f"Spawn ({x:.1f},{y:.1f}) unsafe: min={min_dist:.2f}m < "
        f"{SPAWN_SAFETY_DIST}m, retry {attempt + 1}/{SPAWN_MAX_RETRIES}"
    )
    self.accepting_scans = False
    self.current_scan = np.ones(LIDAR_BEAMS, dtype=np.float32) * LIDAR_MAX_RANGE

return self.get_state()
```

Se tutti e 3 i tentativi falliscono, viene usata l'ultima posizione con warning nel log — il robot potrebbe crashare a step=1, ma questo è un evento raro (spawn list di buona qualità).

---

### Componente 2: Spawn list Maze 2 espansa (8 → 16 punti)

#### Problema della lista attuale

```
(-6.0,  0.0,  0.0 )   # entrance
(-6.0, -1.5,  0.0 )   # entrance -1.5m  ←  stesso corridoio A1
(-6.0,  2.0,  0.0 )   # entrance +2m    ←  stesso corridoio A1
(-6.0,  0.0,  1.57)   # entrance yaw N  ←  stessa posizione A1, solo yaw diverso
(-3.5,  0.5,  0.0 )   # centre-left
(-3.5, -2.5,  1.57)   # WARNING — margine 0.25-0.40m
(-1.5, -2.5,  0.0 )   # centre
( 1.5,  0.0,  3.14)   # right
```

Problemi:
- 4/8 spawn in zona identica (x=-6, ingresso)
- 0 spawn in zona superiore (y > 2m)
- 0 spawn in zona inferiore (y < -3m)
- Solo 3 yaw diversi: 0°, 90°, 180° — nessuna orientazione diagonale

#### Geometria Maze 2 (da labirinto_9b.world)

Muri principali e loro posizione (centro wall, coordinate assolute):

| Wall | x | y | yaw (rad) | Orientazione |
|------|---|---|-----------|-------------|
| Wall_0 | -7.62 | 3.95 | 1.571 | verticale — confine sinistro superiore |
| Wall_14 | -5.29 | -3.36 | 1.608 | verticale — confine sinistro inferiore |
| Wall_15 | -4.95 | -1.80 | 1.011 | diagonale — corridoio interno |
| Wall_17 | -6.37 | 1.62 | -0.262 | diagonale — corridoio ingresso |
| Wall_19 | -3.09 | -1.83 | 1.047 | diagonale — zona centro |
| Wall_20 | -2.75 | -0.24 | 1.551 | quasi verticale — separatore centro |
| Wall_32 | -0.53 | -1.09 | -1.571 | verticale — separatore centro-destra |
| Wall_9 | 1.01 | -2.48 | -1.565 | verticale — destra inferiore |
| Wall_31 | 1.04 | 0.84 | 2.958 | diagonale — destra |
| Wall_30 | 2.60 | 1.77 | -1.571 | verticale — destra superiore |
| Wall_8 | 2.68 | -1.15 | 2.974 | diagonale — destra inferiore |
| Wall_7 | 4.35 | -0.08 | -1.571 | verticale — estremo destra |
| Wall_5 | 7.28 | 3.90 | -1.567 | verticale — confine destra |

Estensione labirinto: x ∈ [-7.6, +7.3], y ∈ [-6.3, +6.5]

#### Nuova spawn list — 16 punti in 6 zone

**Regola di design:** ogni punto deve avere min_lidar ≥ 0.50m in tutte le direzioni (buffer di 0.25m sopra il collision threshold). Validare **tutti** con `test_spawns.sh` prima del training.

```python
SPAWN_LISTS = {
    1: [
        # Maze 1 — invariato
        (-3.0, -5.0,  1.571),  # M1-A1
        ( 0.0, -4.5,  1.571),  # M1-A2
        ( 2.5, -5.0,  1.571),  # M1-A3
        (-1.5, -5.0,  0.0  ),  # M1-A4
        (-2.9, -2.0,  1.571),  # M1-B1
        (-2.9,  0.5,  0.0  ),  # M1-B2
        ( 2.5, -2.0,  1.571),  # M1-C1
        ( 0.5, -2.5,  1.571),  # M1-D1
    ],
    2: [
        # Zone A: ingresso sinistro (2 spawn — ridotti da 4)
        (-6.0,  0.0,  0.0  ),  # A1: heading E (validato OK)
        (-6.0,  2.0,  4.712),  # A2: heading S (nuova orientazione)

        # Zone B: centro-sinistra (3 spawn)
        (-4.5,  0.5,  0.0  ),  # B1: heading E
        (-4.5, -1.5,  1.571),  # B2: heading N (sostituisce WARNING B2)
        (-5.0,  1.5,  2.356),  # B3: heading NW (135°)

        # Zone C: centro (3 spawn)
        (-2.5,  1.0,  0.0  ),  # C1: heading E
        (-1.5, -2.5,  0.0  ),  # C2: heading E (validato OK, era C1)
        (-2.0, -1.0,  0.785),  # C3: heading NE (45°)

        # Zone D: centro-destra (3 spawn)
        ( 1.5,  0.0,  3.142),  # D1: heading W (validato OK)
        ( 1.5, -1.5,  1.571),  # D2: heading N
        ( 2.0,  1.0,  4.712),  # D3: heading S

        # Zone E: superiore (2 spawn — attualmente ZERO copertura)
        (-3.0,  3.0,  0.0  ),  # E1: heading E
        ( 0.0,  3.5,  3.142),  # E2: heading W

        # Zone F: inferiore (3 spawn — attualmente ZERO copertura)
        (-4.5, -3.5,  0.0  ),  # F1: heading E
        (-1.5, -4.0,  1.571),  # F2: heading N
        ( 0.5, -3.5,  3.142),  # F3: heading W
    ],
}
```

**Orientazioni coperte:** 0° (E), 45° (NE), 90° (N), 135° (NW), 180° (W), 270° (S) — 6 su 8 possibili.

**Note sui punti nuovi (da validare con test_spawns.sh):**
- B1 (-4.5, 0.5): spostato da (-3.5, 0.5) verso sx, tra Wall_15 e Wall_17
- B2 (-4.5, -1.5): sostituisce WARNING a (-3.5, -2.5), più lontano da Wall_15 e Wall_19
- B3 (-5.0, 1.5): corridoio superiore sinistro, tra Wall_0 e Wall_17
- C1 (-2.5, 1.0): zona centro, tra Wall_20 e Wall_21
- C3 (-2.0, -1.0): zona centro-inferiore, tra Wall_20 e Wall_32
- D2 (1.5, -1.5): tra Wall_9 e Wall_32
- D3 (2.0, 1.0): tra Wall_31 e Wall_30
- E1 (-3.0, 3.0): zona superiore sinistra, above Wall_21 e Wall_22
- E2 (0.0, 3.5): zona superiore destra, below Wall_3 e Wall_26
- F1 (-4.5, -3.5): zona inferiore sinistra, below Wall_14, above Wall_13
- F2 (-1.5, -4.0): zona inferiore centro, below Wall_19/Wall_20
- F3 (0.5, -3.5): zona inferiore destra, between Wall_10 e Wall_9

---

## 3. File da modificare

| File | Modifica |
|------|---------|
| `src/my_usv/scripts/usv_env.py` | 1) Costanti `SPAWN_SAFETY_DIST`, `SPAWN_MAX_RETRIES`; 2) `reset_environment` con retry loop; 3) `SPAWN_LISTS[2]` sostituita |

Nessun altro file cambia.

---

## 4. Procedura di validazione (OBBLIGATORIA prima del training)

```bash
# Checkout feng_direct
git checkout feng_direct

# 1. Build (propaga spawn list aggiornata a install/)
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project \
    bash -c "cd /home/usv_ws && colcon build --packages-select my_usv"

# 2. Valida TUTTI i 16 spawn su Maze 2
./test_spawns.sh 2
```

**Criteri di accettazione:**
- Tutti i punti nuovi devono essere ✅ OK (min LIDAR > 0.40m)
- Nessun ❌ COLLISION
- Per ogni ⚠️ WARNING: spostare il punto di 0.3–0.5m nella direzione libera e ri-validare

---

## 5. Test del safety check

Dopo l'implementazione, aggiungere in `test_usv_logic.py` (o script separato) un test che verifica:
- `reset_environment` con spawn list contenente un punto volutamente vicino a un muro (mock `current_scan`) invoca il retry
- Dopo 3 fallimenti usa l'ultima posizione disponibile senza crash del processo

---

## 6. Perché questo migliora la generalizzazione a Maze 3

Il robot che vede solo la zona ingresso di Maze 2 (x≈-6) impara "naviga dal corridoio sinistro". Con 16 spawn distribuiti su tutto il maze:
- Impara comportamenti LIDAR-locali: come evitare muri diagonali, come gestire corridoi stretti, come navigare in spazio aperto
- Le zone E e F (superiore/inferiore) espongono configurazioni di muri mai viste dal punto di ingresso
- Yaw diagonale (45°, 135°) forza il robot a imparare da orientazioni che coincidono con i muri diagonali di Maze 2 e Maze 3

Questi pattern LIDAR locali sono trasferibili a maze non visti (Maze 1 test set, eventuali maze futuri).

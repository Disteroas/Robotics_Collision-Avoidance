# Reward Function & Curriculum Learning Fix — Design Spec

**Date:** 2026-05-04
**Branch:** prova_claude_code
**Status:** Approved for implementation

---

## Problem Summary

Analisi dei dati di training (`risultati/multi_maze_05_01/`) conferma tre problemi critici:

1. **Reward hacking**: ultimi 100 ep su maze 1/2 → tutti esattamente 500 steps, 0 crash. Il robot sopravvive girando in cerchio, non navigando. La reward attuale (+5/step) premia la sopravvivenza pura senza richiedere comportamento utile.

2. **Catastrophic forgetting**: oscillazione avg100 di ±250 ogni switch di maze (ep 2000–3000). Firma classica del block-alternation curriculum.

3. **Zero generalizzazione**: maze 3 (unseen) → 100% crash rate. Due failure mode distinti (~80 steps e ~414 steps) indicano overfitting a geometrie specifiche di maze 1/2.

4. **MAX_STEPS troppo basso**: 500 steps = 25m percorsi. Tetto raggiunto sistematicamente → impossibile distinguere "naviga bene" da "sopravvive immobile".

---

## Decisioni di Design

### 1. Reward Function

**Rimuovi:** steering penalty `-0.1 × |action-5|`
**Sostituisci con:** soft regularization `-0.02 × |action-5|` (anti-oscillazione in spazio aperto)

**Aggiungi:** open-space bonus `+SPACE_BONUS_WEIGHT × mean(scan) / LIDAR_MAX_RANGE`
- `mean(scan)` ∈ [0, 5.0] → normalizzato in [0, 1]
- `SPACE_BONUS_WEIGHT = 2.0` → bonus max +2.0 in spazio completamente libero
- Incentiva il robot a cercare zone aperte, penalizza implicitamente lo stare vicino ai muri
- Risolve lo spinning: girare in cerchio vicino a muri abbassa mean(scan) → reward inferiore

**Estendi zona pericolo frontale:**
- `FRONT_DANGER`: 1.5m → **3.0m** (robot vede il muro 15 step prima a 0.5 m/s)
- Esponente front penalty: cubico → **quadratico** (segnale più forte a distanza media)
  - A 2.0m dal muro: penalità cubica ≈ 0.22 → penalità quadratica ≈ 0.99 (4.5× più forte)
  - Risolve il problema "mare aperto → sbatte al primo muro": segnale di pericolo anticipa la collisione

**SIDE_DANGER:** invariato (0.45m, quadratico)

**Reward completa post-fix:**
```python
# Collisione
if min_dist < COLLISION_DIST:
    return -1000.0, True

# Base
reward = 5.0

# Open-space bonus (nuovo)
space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE

# Steering regularization (ridotto da 0.1 a 0.02)
steering_penalty = abs(action_index - 5) * 0.02

# Front danger (quadratico, zona estesa a 3.0m)
if front_dist < FRONT_DANGER:
    severity = (FRONT_DANGER - front_dist) / (FRONT_DANGER - COLLISION_DIST)
    danger_penalty += 20.0 * (severity ** 2)

# Side danger (invariato)
if right_dist < SIDE_DANGER:
    severity = (SIDE_DANGER - right_dist) / (SIDE_DANGER - COLLISION_DIST)
    danger_penalty += 5.0 * (severity ** 2)

if left_dist < SIDE_DANGER:
    severity = (SIDE_DANGER - left_dist) / (SIDE_DANGER - COLLISION_DIST)
    danger_penalty += 5.0 * (severity ** 2)

return reward + space_bonus - steering_penalty - danger_penalty, False
```

**Nuove costanti in `usv_logic.py`:**
```python
FRONT_DANGER       = 3.0   # era 1.5
SPACE_BONUS_WEIGHT = 2.0   # nuovo
```

---

### 2. MAX_STEPS

`MAX_STEPS`: 500 → **1000**

Modifica in `train.py`. Nessun impatto su architettura o reward.

---

### 3. Curriculum Learning — Progressivo con Threshold

**Sostituisce:** alternanza a blocchi fissi (maze 1 / maze 2 ogni 100 ep)

**Phase 1 — Solo Maze 1:**
- Training esclusivo su maze 1 finché `avg100_maze1 > 1500` misurata su finestra 50 episodi
- Basandosi sui dati storici, threshold raggiunta intorno a ep 1200–1400
- `train.py` scrive `phase.txt` con valore `"2"` quando threshold scattata

**Phase 2 — Mixed 30/70:**
- Ogni nuovo blocco: maze 1 con probabilità 0.30, maze 2 con probabilità 0.70
- Selezione casuale per blocco (non per episodio) — richiede restart Gazebo tra blocchi, non mid-episode
- Il replay buffer di Phase 1 resta intatto → mitigazione forgetting naturale

**Meccanismo di comunicazione train.py → bash:**
```
src/my_usv/scripts/phase.txt  →  "1" o "2"
```
`start_training_curriculum.sh` legge questo file prima di ogni blocco per scegliere il maze.

**Threshold detection in `train.py`:**
```python
# Alla fine di ogni episodio maze 1
if maze_id == 1 and len(maze1_recent) >= 50:
    if np.mean(maze1_recent[-50:]) > PHASE2_THRESHOLD:
        write_phase_file("2")
```

**Nuova costante in `train.py`:**
```python
PHASE2_THRESHOLD = 1500   # avg100 su finestra 50 ep maze 1
```

---

## File Modificati

| File | Tipo | Cambiamento |
|---|---|---|
| `src/my_usv/scripts/usv_logic.py` | Modifica | `FRONT_DANGER` 1.5→3.0, aggiungi `SPACE_BONUS_WEIGHT`, aggiorna `compute_reward()` |
| `src/my_usv/scripts/train.py` | Modifica | `MAX_STEPS` 500→1000, aggiungi threshold detection, scrittura `phase.txt` |
| `start_training_curriculum.sh` | Modifica | Rimuovi alternanza fissa, leggi `phase.txt`, Phase 1 = solo maze 1, Phase 2 = 30/70 random per blocco |
| `src/my_usv/test/test_usv_logic.py` | Modifica | Aggiorna test per nuovi valori `FRONT_DANGER` e `SPACE_BONUS_WEIGHT` |

---

## Test da Aggiornare / Aggiungere

I test esistenti in `test_usv_logic.py` che usano `FRONT_DANGER` o il valore cubico vanno aggiornati:
- `test_front_danger_severity_is_cubic` → rinominare e aggiornare a quadratico
- `test_front_danger_reduces_reward` → midpoint cambia con nuovo `FRONT_DANGER=3.0`
- Aggiungere `test_space_bonus_increases_with_distance_from_walls`
- Aggiungere `test_space_bonus_zero_when_all_walls_at_collision_dist`

---

## Metriche di Successo

Training considerato migliorato se, dopo 3000 ep con nuovo setup:
- Maze 1 crash rate ultimi 100 ep < 5% (come ora)
- Maze 2 crash rate ultimi 100 ep < 10% (ora 1%, non regredire)
- **Maze 3 crash rate < 50%** (ora 100% — qualsiasi miglioramento è progresso)
- avg100 finale > 3000 (scala cambia con MAX_STEPS=1000 e space bonus)
- Nessuna oscillazione >±500 avg100 tra blocchi consecutivi (forgetting ridotto)

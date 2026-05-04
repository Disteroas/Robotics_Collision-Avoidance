# Modifiche apportate da Claude Code — branch `prova_claude_code`

## Contesto

Partendo da `multi_maze_train` (commit `68f9824`), questo branch applica tre categorie di modifiche:
code review fixes, refactoring per testabilità, e suite di test TDD completa.

---

## 1. Code review fixes (`usv_env.py`)

### Branch `else` morto rimosso
`_scan_cb` aveva un guard invertito:
```python
if len(raw_scan) != LIDAR_BEAMS:   # sempre True con URDF a 512 ray
    ...  # min-pooling
else:
    self.current_scan = raw_scan   # dead code, mai raggiunto
```
Rimosso il branch `else`. Min-pooling ora gira incondizionatamente per qualsiasi
valore di `LIDAR_BEAMS`, eliminando una bomba silenziosa.

### Reset `_lidar_checked` ad ogni restart Gazebo
`_lidar_checked = False` aggiunto in `reset_environment()`. Prima il log di
validazione beam count spariva dopo il primo episodio. Ora riappare ad ogni
blocco (30 restart totali in un training completo), utile per diagnosticare
configurazioni errate del sensore.

### Commento settori angolari
Aggiunto in `_compute_reward`:
```python
# FOV 270° / 50 bin = 5.4°/bin → destra [-135°,-54°], fronte [-54°,+54°], sinistra [+54°,+135°]
```
I magic indices `[0:15]`, `[15:35]`, `[35:50]` ora hanno una motivazione
geometrica esplicita.

---

## 2. Refactoring per testabilità

### Problema
Tutta la logica pura (LIDAR processing, reward) era dentro `UsvEnv(Node)`.
Istanziare `UsvEnv` richiede `rclpy.init()` + Gazebo attivo → impossibile
in un test unitario. TDD dice: se il test è difficile da scrivere, il design
è sbagliato.

### `usv_logic.py` (nuovo file)
Estratte due funzioni pure da `usv_env.py`:
- `process_lidar(raw_ranges, n_bins, max_range)` — NaN handling + clip + min-pooling
- `compute_reward(scan, action_index)` — collision check + steering + danger zones

Tutte le costanti (`LIDAR_MAX_RANGE`, `LIDAR_BEAMS`, `COLLISION_DIST`,
`FRONT_DANGER`, `SIDE_DANGER`, `LINEAR_VEL`) spostate qui come unica sorgente
di verità.

### `train_core.py` (nuovo file)
Estratti da `train.py` senza import ROS2:
- `ReplayBuffer`
- `DDQNAgent`
- `save_ckpt` / `load_ckpt`
- Tutte le costanti di training (`GAMMA`, `LR`, `BATCH_SIZE`, ecc.)

`train.py` diventa un thin wrapper: solo `main()`, CLI parsing, e glue ROS2.

### `usv_env.py` aggiornato
- Importa `process_lidar`, `compute_reward` da `usv_logic`
- `_scan_cb` chiama `process_lidar(msg.ranges)` invece di avere la logica inline
- `step_action` chiama `compute_reward` direttamente
- Rimosso metodo `_compute_reward` (era wrapper ridondante)
- Rimossa ridefinizione locale delle costanti

---

## 3. Suite di test TDD

### Struttura
```
src/my_usv/test/
├── conftest.py              # aggiunge scripts/ a sys.path
├── test_usv_logic.py        # 14 test: LIDAR processing + reward
├── test_ddqn_model.py       # 6 test: forward pass, shape, gradienti
├── test_replay_buffer.py    # 6 test: push, capacity, sample shape
└── test_agent.py            # 9 test: epsilon, greedy, learn, target sync
```

**Totale: 35 test → tutti GREEN (verificati in Docker)**

### `test_usv_logic.py` — cosa testa
| Test | Comportamento verificato |
|---|---|
| `test_output_is_exactly_50_bins` | Output sempre 50 elementi |
| `test_nan_replaced_with_max_range` | NaN → 5.0m |
| `test_pos_inf_replaced_with_max_range` | +inf → 5.0m |
| `test_values_above_max_clipped` | Valori > 5.0 clampati |
| `test_all_output_values_in_valid_range` | Output sempre in [0, 5.0] |
| `test_min_pooling_picks_nearest_obstacle_in_bin` | Ostacolo singolo preservato nel bin corretto |
| `test_obstacle_in_last_bin_detected` | Ostacolo all'ultimo raggio → ultimo bin |
| `test_collision_returns_minus_1000_and_done` | Collisione = -1000, done=True |
| `test_collision_triggered_by_single_ray_below_threshold` | Basta un raggio < 0.25m |
| `test_clear_path_straight_returns_base_reward` | Percorso libero + dritto = +5.0 esatto |
| `test_hard_left_has_steering_penalty` | Action 0 → reward 4.5 (penalità 0.5) |
| `test_hard_right_has_same_penalty_as_hard_left` | Penalità simmetrica |
| `test_front_danger_severity_is_cubic` | Penalità fronte = 20×severity³ |
| `test_side_danger_severity_is_quadratic` | Penalità lato = 5×severity² |

### Come eseguire
```bash
# Con Docker Desktop aperto e progetto compilato:
docker run --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash -c \
  "cd /home/usv_ws && python3 -m pytest src/my_usv/test/ -v"
```

### `Dockerfile` aggiornato
Aggiunto `RUN pip3 install pytest` per rendere i test eseguibili nel container
senza installazione manuale.

# Modifiche apportate da Claude Code — branch `prova_claude_code`

## Contesto

Partendo da `multi_maze_train` (commit `68f9824`), questo branch applica cinque categorie di modifiche:
code review fixes, refactoring per testabilità, suite di test TDD completa, fix reward function, e curriculum learning progressivo.

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

---

## 8. Gymnasium wrapper (`gym_env_claude` branch)

**Obiettivo:** esporre `UsvEnv` come `gymnasium.Env` standard per poter pluggare qualsiasi algoritmo DRL (XinJingHao, stable-baselines3, cleanRL) senza riscrivere glue code.

### Pattern: composizione

`UsvGymEnv(gymnasium.Env)` contiene `UsvEnv` come attributo. Zero modifiche a `usv_env.py`, `train.py`, `train_core.py`. Il lifecycle ROS2 (`rclpy.init/shutdown`) è gestito interamente nel wrapper.

### `usv_gym_env.py` (nuovo file)

| Parametro costruttore | Default | Descrizione |
|---|---|---|
| `continuous` | `False` | `False` → `Discrete(11)`, `True` → `Box(-0.8, 0.8, shape=(1,))` |
| `max_steps` | `1000` | Limite step per episodio (indipendente da `train.py`) |

**API gymnasium:**
- `reset(seed, options)` → `(obs, {})`
- `step(action)` → `(obs, float(reward), terminated, truncated, info)`
- `close()` → `destroy_node()` + `rclpy.shutdown()`

**Distinzione critica `terminated` vs `truncated`:**
- `terminated=True` → collisione reale. Bootstrap valore = 0.
- `truncated=True` → step limit raggiunto senza collisione. Bootstrap valore = `γ·V(s')`.

Passare `terminated` (non `done`) al replay buffer elimina bias sistematico su ogni episodio troncato.

**Mapping azione continua→indice discreto:**
```python
idx = int(np.clip(round((float(action[0]) + 0.8) / 0.16), 0, 10))
```
Formula inversa di `angular_z = -0.8 + 0.16 * idx`.

### `train_gym.py` (nuovo file)

Entry point training via gymnasium. Differenze da `train.py`:
- Usa `UsvGymEnv` invece di `UsvEnv` direttamente
- Singolo maze, nessun curriculum (per validare il wrapper)
- Checkpoint separato: `checkpoint_gym.pth` (non sovrascrive `checkpoint.pkl`)
- Swap point documentato: sostituisci 1 import per cambiare algoritmo

```python
# ── swap point ──────────────────────────────────────────────────────
# from xjh_ddqn import DQN_Agent as DDQNAgent   # XinJingHao DDQN
# from xjh_ppo   import PPO_Agent as DDQNAgent   # XinJingHao PPO
# ────────────────────────────────────────────────────────────────────
```

### Test aggiunti (`test_usv_gym_env.py`)

11 test, `UsvEnv` completamente mockato (nessun ROS2 richiesto):

| Test | Comportamento |
|---|---|
| `test_observation_space_shape` | `Box(0,1,shape=(50,),float32)` |
| `test_action_space_discrete` | `Discrete(11)` |
| `test_action_space_continuous` | `Box(-0.8,0.8,shape=(1,))` |
| `test_reset_returns_obs_and_empty_info` | `reset()` → `(ndarray, {})` |
| `test_reset_resets_step_counter` | step counter azzerato dopo reset |
| `test_step_returns_correct_5_tuple` | 5-tuple con tipi corretti |
| `test_terminated_true_on_crash` | crash → `terminated=True, truncated=False` |
| `test_truncated_true_on_step_limit` | step limit → `truncated=True, terminated=False` |
| `test_terminated_false_on_truncation` | truncation NON azzera terminated |
| `test_continuous_action_maps_center_to_index_5` | `0.0` → `idx=5` |
| `test_continuous_action_maps_extremes` | `±0.8` → `idx=0/10` |

**Suite totale: 52 test GREEN**

### `Dockerfile` aggiornato

```dockerfile
RUN pip3 install pytest gymnasium
```

Commits: `c57797b` (Dockerfile), `a4d456b` (tests RED), `73815a5` (impl GREEN), `f6c00d1` (train_gym.py)

---

## 4. Fix reward function (`usv_logic.py`)

**Problema diagnosticato:** analisi dei log `risultati/multi_maze_05_01/` mostrava reward hacking (robot gira in cerchio per sopravvivere), zero generalizzazione su maze 3, e crash al primo muro dopo zone aperte.

### Modifiche a `usv_logic.py`

**Nuove costanti:**
```python
FRONT_DANGER       = 3.0    # era 1.5m — robot vede muro 15 step prima a 0.5 m/s
SPACE_BONUS_WEIGHT = 2.0    # nuovo — bonus max in spazio completamente libero
```

**Open-space bonus (nuovo):**
```python
space_bonus = SPACE_BONUS_WEIGHT * float(np.mean(scan)) / LIDAR_MAX_RANGE
```
`mean(scan)` ∈ [0, 5.0] → normalizzato in [0, 1] → bonus max +2.0. Risolve lo spinning: girare vicino ai muri abbassa `mean(scan)` → reward inferiore.

**Steering penalty ridotta:**
```python
steering_penalty = abs(action_index - 5) * 0.02   # era 0.1
```
Meno vincolo comportamentale, più anti-oscillazione in spazio aperto.

**Front danger: cubico → quadratico su zona estesa:**
```python
# era: 20.0 * (severity ** 3), FRONT_DANGER=1.5
danger_penalty += 20.0 * (severity ** 2)   # FRONT_DANGER=3.0
```
A 2.0m dal muro: penalità cubica ≈ 0.22 → quadratica ≈ 0.99 (4.5× più forte). Risolve "mare aperto → sbatte al primo muro".

**Reward completa post-fix:**
```python
return 5.0 + space_bonus - steering_penalty - danger_penalty, False
```

### Modifiche a `test_usv_logic.py`

Test aggiornati (valori attesi cambiati con nuova formula):
- `test_clear_path_straight_returns_base_reward`: 5.0 → **7.0** (+2.0 space bonus)
- `test_hard_left_has_steering_penalty`: 4.5 → **6.9**
- `test_front_danger_severity_is_cubic` → rinominato **`test_front_danger_severity_is_quadratic`**
- `test_front_danger_reduces_reward`: scan fronte aggiornato a 2.0m (< nuovo FRONT_DANGER=3.0)
- 4 test side-danger: expected aggiornati per includere space_bonus

Nuovi test aggiunti:
| Test | Comportamento verificato |
|---|---|
| `test_space_bonus_increases_with_open_space` | scan mean=5.0 → reward > scan mean=4.0 (no danger zone triggered) |
| `test_steering_penalty_reduced_to_0_02` | hard turn penalty = 0.1 esatto |

**Totale suite post-fix: 41 test → tutti GREEN**

Commits: `7bbbf7a` (reward impl), `b5c6462` (fix false-positive test)

---

## 5. MAX_STEPS 500 → 1000 (`train.py`)

**Problema:** 500 step = 25m percorsi. Tetto raggiunto sistematicamente → impossibile distinguere "naviga bene" da "sopravvive immobile".

```python
MAX_STEPS = 1000   # era 500
```

Commit: `8dc79c0`

---

## 6. Phase detection curriculum (`train.py`)

**Problema:** il curriculum a blocchi fissi (maze1/maze2 ogni 100 ep) causava catastrophic forgetting (oscillazione avg100 ±250 ad ogni switch).

### Nuove costanti
```python
PHASE2_THRESHOLD = 1500   # avg reward maze1 su finestra 50 ep per passare a Phase 2
PHASE1_WINDOW    = 50     # dimensione finestra rolling
```

### Nuovo argomento CLI
```bash
--phase-file src/my_usv/scripts/phase.txt   # default
```

### Funzione `_write_phase()`
```python
def _write_phase(phase_file: str, phase: int) -> None:
    with open(phase_file, 'w') as f:
        f.write(str(phase))
```

### Rilevamento threshold nel loop episodico
Dopo `rh.append(ep_rew)`, solo per maze 1:
```python
if args.maze_id == 1:
    maze1_window.append(ep_rew)
    if (len(maze1_window) == PHASE1_WINDOW
            and float(np.mean(maze1_window)) > PHASE2_THRESHOLD):
        phase_path = os.path.abspath(args.phase_file)
        if not os.path.exists(phase_path) or open(phase_path).read().strip() == '1':
            _write_phase(phase_path, 2)
```

La guardia `== '1'` evita scritture ripetute dopo lo switch. `phase.txt` contiene `"1"` o `"2"` e viene letto dallo script bash ad ogni blocco.

Commit: `26ebf0b`

---

## 7. Curriculum progressivo (`start_training_curriculum.sh`)

**Rimosso:** alternanza fissa `MAZE_SEQUENCE=(1 2)` con logica `block % NUM_MAZES`.

**Aggiunto:**
```bash
PHASE_FILE="${SCRIPTS_HOST}/phase.txt"
PHASE2_PROB=70   # % probabilità maze 2 in Phase 2
```

**Nuova funzione `select_maze()`:**
```bash
select_maze() {
    local phase=1
    if [ -f "$PHASE_FILE" ]; then
        phase=$(cat "$PHASE_FILE" | tr -d '[:space:]')
    fi
    if [ "$phase" = "2" ]; then
        local roll=$(( RANDOM % 100 ))
        if [ "$roll" -lt "$PHASE2_PROB" ]; then echo 2; else echo 1; fi
    else
        echo 1   # Phase 1: sempre maze 1
    fi
}
```

**Logica risultante:**
- **Phase 1** (default): training esclusivo su maze 1 finché `avg50_maze1 > 1500`
- **Phase 2** (dopo threshold): 70% maze 2 / 30% maze 1 per blocco (selezione casuale per blocco, non per episodio)

**`--reset` aggiornato** per cancellare anche `phase.txt`.

**`--phase-file`** passato a `train.py` in `run_train_block()`.

Commit: `766a2c6`

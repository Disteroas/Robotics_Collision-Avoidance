# Gymnasium Wrapper — Design Spec

**Date:** 2026-05-05
**Branch:** gym_env_claude
**Status:** Approved for implementation

---

## Problem Summary

`UsvEnv` espone un'API proprietaria (`reset_environment()`, `step_action()`). Gli algoritmi DRL della repo XinJingHao/DRL-Pytorch (e più in generale qualsiasi libreria moderna: stable-baselines3, cleanRL) aspettano `gymnasium.Env`. Senza wrapper, ogni algoritmo nuovo richiede riscrittura del glue code.

**Obiettivo:** wrapper `UsvGymEnv(gymnasium.Env)` che espone l'interfaccia gymnasium standard, mantenendo `UsvEnv` e `train.py` intatti. Un secondo entry point `train_gym.py` valida il wrapper con DDQN (algoritmo attuale) e diventa il punto di swap per futuri algoritmi.

---

## Decisioni di Design

### 1. Pattern architetturale: composizione

`UsvGymEnv` contiene `UsvEnv` come attributo (`self._env`), non eredita da esso.

**Motivazione:**
- Nessuna modifica a `usv_env.py` (codice battle-tested, curriculum integrato)
- Lifecycle ROS2 (`rclpy.init/shutdown`) gestito interamente dentro il wrapper
- `UsvEnv` rimane testabile indipendentemente dal wrapper gymnasium

### 2. Action space: dual mode via flag costruttore

```python
UsvGymEnv(continuous=False)   # Discrete(11) — identico al DDQN attuale
UsvGymEnv(continuous=True)    # Box(-0.8, 0.8, shape=(1,)) — angular velocity diretta
```

Mapping continuo→discreto in `step()`:
```python
idx = int(np.clip(round((float(action[0]) + 0.8) / 0.16), 0, 10))
```
Formula inversa di `angular_z = -0.8 + 0.16 * idx`. Nessuna modifica a `step_action()`.

### 3. `terminated` vs `truncated` — distinzione obbligatoria

Gymnasium v26+ separa i due casi. Non è opzionale:

- `terminated=True` → collisione (`crashed=True` da `compute_reward`). Valore bootstrap = 0.
- `truncated=True` → step limit raggiunto senza collisione. Valore bootstrap = `γ·V(s')`.

**Impatto sul replay buffer:**
```python
# CORRETTO — passa solo terminated, non done
agent.memory.push(state, action, reward, next_state, terminated)
```
Con `MAX_STEPS=1000` e reward fino a +7/step, un episodio troncato vale ~3000 di valore futuro. Azzerarlo via `done=True` introduce bias sistematico su ogni episodio che raggiunge il limite.

### 4. `max_steps` come parametro costruttore

`max_steps=1000` come default, non importato da `train.py`. Il wrapper deve essere autonomo: `train_gym.py` potrebbe usare un limite diverso da `train.py`.

### 5. `train_gym.py` usa `DDQNAgent` da `train_core.py`

Non si copia il DDQN di XinJingHao perché `DDQNAgent` in `train_core.py` è funzionalmente identico. Il valore è nell'interfaccia gymnasium, non nel duplicare codice. `train_gym.py` include un "swap point" documentato per sostituire l'agente con qualsiasi altro (XinJingHao PPO, SAC, ecc.) cambiando una sola import.

### 6. Curriculum assente in `train_gym.py`

Prima versione: singolo maze, no phase detection. Obiettivo è validare il wrapper. Il curriculum può essere aggiunto successivamente su `train_gym.py` se i risultati lo giustificano.

### 7. Checkpoint separato

`train_gym.py` salva in `checkpoint_gym.pth` (formato PyTorch state dict). Formato diverso da `checkpoint.pkl` di `train.py` — i due training coesistono senza conflitti.

---

## Interfaccia Completa

### `UsvGymEnv`

```python
class UsvGymEnv(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, continuous: bool = False, max_steps: int = 1000):
        super().__init__()
        rclpy.init()
        self._env       = UsvEnv()
        self._cont      = continuous
        self._max_steps = max_steps
        self._steps     = 0

        self.observation_space = Box(
            low=0.0, high=1.0, shape=(LIDAR_BEAMS,), dtype=np.float32
        )
        if continuous:
            self.action_space = Box(
                low=np.float32(-0.8), high=np.float32(0.8),
                shape=(1,), dtype=np.float32
            )
        else:
            self.action_space = Discrete(11)

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._steps = 0
        obs = self._env.reset_environment()
        return obs, {}

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._cont:
            idx = int(np.clip(round((float(action[0]) + 0.8) / 0.16), 0, 10))
        else:
            idx = int(action)

        obs, reward, crashed = self._env.step_action(idx)
        self._steps += 1

        terminated = crashed
        truncated  = (not crashed) and (self._steps >= self._max_steps)

        return obs, float(reward), terminated, truncated, {'steps': self._steps, 'crashed': crashed}

    def close(self) -> None:
        self._env.destroy_node()
        rclpy.shutdown()
```

---

## File Modificati

| File | Tipo | Cambiamento |
|---|---|---|
| `src/my_usv/scripts/usv_gym_env.py` | Nuovo | `UsvGymEnv(gymnasium.Env)` — wrapper completo |
| `src/my_usv/scripts/train_gym.py` | Nuovo | Entry point DDQN via gymnasium, swap point per altri algo |
| `src/my_usv/test/test_usv_gym_env.py` | Nuovo | 10 test TDD, `UsvEnv` mockato |
| `Dockerfile` | Modifica | `pip3 install pytest gymnasium` |

**Nessuna modifica a:** `usv_env.py`, `usv_logic.py`, `train.py`, `train_core.py`, `start_training_curriculum.sh`.

---

## Test da Implementare

| Test | Comportamento verificato |
|---|---|
| `test_observation_space_shape` | `Box(0, 1, shape=(50,), float32)` |
| `test_action_space_discrete` | `Discrete(11)` con `continuous=False` |
| `test_action_space_continuous` | `Box(-0.8, 0.8, shape=(1,))` con `continuous=True` |
| `test_reset_returns_obs_and_empty_info` | `reset()` → `(ndarray shape (50,), {})` |
| `test_step_returns_5_tuple` | `step()` → tuple di lunghezza 5 con tipi corretti |
| `test_terminated_true_on_crash` | `crashed=True` → `terminated=True, truncated=False` |
| `test_truncated_true_on_step_limit` | step count = max_steps, no crash → `truncated=True, terminated=False` |
| `test_terminated_false_on_truncation` | `truncated=True` implica `terminated=False` (distinzione critica) |
| `test_continuous_action_maps_center_to_index_5` | `action=0.0` → `idx=5` |
| `test_continuous_action_maps_extremes` | `action=-0.8` → `idx=0`, `action=0.8` → `idx=10` |

---

## Metriche di Successo

- Tutti i 10 test nuovi GREEN in Docker
- Suite completa (51 test) GREEN — nessuna regressione
- `train_gym.py --maze-id 1 --episodes 10` gira senza errori in Docker con Gazebo attivo
- Dopo 100 episodi con `train_gym.py`: comportamento comparabile a `train.py` sullo stesso maze (avg100 nella stessa scala)

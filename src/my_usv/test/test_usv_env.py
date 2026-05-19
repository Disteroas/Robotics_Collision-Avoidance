import sys, os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from usv_logic import LIDAR_MAX_RANGE, LIDAR_BEAMS

# Import constants from usv_env module-level (not the ROS node class)
# We need to import just the module-level data, not instantiate UsvEnv
import importlib, types

def _load_env_module():
    """Load usv_env without triggering ROS2 node init."""
    import importlib.util

    # Helper to create a stub module with arbitrary attribute access
    def _stub_module(name):
        m = types.ModuleType(name)
        # Any attribute access returns a trivial class
        class _AnyAttr:
            def __getattr__(self, n):
                return type(n, (), {})()
        m.__getattr__ = lambda n: type(n, (), {})
        return m

    # Build stub modules for every ROS dependency
    ros_mods = [
        'rclpy', 'rclpy.node', 'rclpy.parameter', 'rclpy.time',
        'geometry_msgs', 'geometry_msgs.msg',
        'nav_msgs', 'nav_msgs.msg',
        'sensor_msgs', 'sensor_msgs.msg',
        'std_srvs', 'std_srvs.srv',
        'gazebo_msgs', 'gazebo_msgs.srv',
    ]
    for mod_name in ros_mods:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = _stub_module(mod_name)

    # Ensure submodule attributes exist on parent modules
    import sys as _sys

    def _ensure_attr(parent_name, attr, child_name):
        parent = _sys.modules[parent_name]
        if not hasattr(parent, attr):
            setattr(parent, attr, _sys.modules[child_name])

    _ensure_attr('rclpy', 'node', 'rclpy.node')
    _ensure_attr('rclpy', 'parameter', 'rclpy.parameter')
    _ensure_attr('rclpy', 'time', 'rclpy.time')
    _ensure_attr('geometry_msgs', 'msg', 'geometry_msgs.msg')
    _ensure_attr('sensor_msgs', 'msg', 'sensor_msgs.msg')
    _ensure_attr('std_srvs', 'srv', 'std_srvs.srv')
    _ensure_attr('gazebo_msgs', 'srv', 'gazebo_msgs.srv')

    # Provide specific stub classes that usv_env imports by name
    def _stub_class(name):
        return type(name, (), {'__init__': lambda self, *a, **k: None})

    setattr(_sys.modules['geometry_msgs.msg'], 'Twist', _stub_class('Twist'))
    setattr(_sys.modules['sensor_msgs.msg'], 'LaserScan', _stub_class('LaserScan'))
    setattr(_sys.modules['std_srvs.srv'], 'Empty', _stub_class('Empty'))
    setattr(_sys.modules['gazebo_msgs.srv'], 'SetEntityState', _stub_class('SetEntityState'))

    # Node stub
    node_mod = _sys.modules['rclpy.node']
    if not hasattr(node_mod, 'Node'):
        class Node:
            def __init__(self, *a, **k): pass
        node_mod.Node = Node

    # Parameter stub on both rclpy and rclpy.parameter
    param_mod = _sys.modules['rclpy.parameter']
    if not hasattr(param_mod, 'Parameter'):
        class Parameter:
            class Type:
                BOOL = 1
        param_mod.Parameter = Parameter
    rclpy_mod = _sys.modules['rclpy']
    rclpy_mod.Parameter = param_mod.Parameter

    # Load the actual module
    spec = importlib.util.spec_from_file_location(
        'usv_env',
        os.path.join(os.path.dirname(__file__), '..', 'scripts', 'usv_env.py')
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_env = _load_env_module()
SPAWN_LISTS = _env.SPAWN_LISTS


def test_maze2_spawn_count_at_least_4():
    # TODO: nuovo maze M2 (ddqn_enhanced_18_05) richiede re-validazione completa spawn.
    # Lista attuale: 6 punti validati su vecchio maze. Espandere dopo validazione visiva.
    assert len(SPAWN_LISTS[2]) >= 4


def test_maze2_all_spawn_entries_are_three_floats():
    for entry in SPAWN_LISTS[2]:
        assert len(entry) == 3, f"Entry {entry} should have 3 elements"
        for val in entry:
            assert isinstance(val, float), f"Value {val} in {entry} should be float"


def test_maze2_spawn_covers_minimum_zones():
    """Almeno A, D, F coperti — Zone B/C/E vuote in lista merge16_05 (da re-validare sul nuovo maze)."""
    spawns = SPAWN_LISTS[2]
    zones = {
        'A': [s for s in spawns if s[0] <= -5.5],               # x <= -5.5
        'D': [s for s in spawns if s[0] > -1.0],                # destra
        'F': [s for s in spawns if s[1] <= -3.0],               # bassa
    }
    for name, pts in zones.items():
        assert len(pts) >= 1, f"Zone {name} has no spawn points"


def test_maze2_yaws_in_valid_range():
    """Tutti i yaw devono essere in [0, 2π)."""
    for entry in SPAWN_LISTS[2]:
        yaw = entry[2]
        assert 0.0 <= yaw < 2 * math.pi + 0.01, f"Yaw {yaw} fuori range [0, 2π)"


def test_spawn_safety_dist_is_0_40():
    assert _env.SPAWN_SAFETY_DIST == 0.40


def test_spawn_max_retries_is_3():
    assert _env.SPAWN_MAX_RETRIES == 3


def test_d1_new_position_zone_d_right():
    """D1 deve essere ricollocato in zona D centro-destra con heading N.

    Posizione originale (1.5, 0.0, π=W) era unfair: heading W spingeva il robot
    in muro centrale entro 60 step. Nuova posizione (3.5, -0.5, π/2=N) ha
    clearance 0.96m e corridoio aperto a nord.

    Vedi analisi_maze/maze2_geom_check.py per analisi geometrica completa.
    """
    spawns_2 = SPAWN_LISTS[2]
    # Cerca la voce D1 (qualsiasi voce con x in [3.0, 4.0] e y in [-1.0, 0.0])
    d_candidates = [s for s in spawns_2 if 3.0 <= s[0] <= 4.0 and -1.0 <= s[1] <= 0.0]
    assert len(d_candidates) == 1, (
        f"Atteso esattamente 1 spawn in zona D centro-destra, trovati {len(d_candidates)}: {d_candidates}"
    )
    x, y, yaw = d_candidates[0]
    assert x == 3.5, f"D1 x atteso 3.5, got {x}"
    assert y == -0.5, f"D1 y atteso -0.5, got {y}"
    assert abs(yaw - math.pi/2) < 0.01, f"D1 yaw atteso π/2 ({math.pi/2:.4f}), got {yaw}"


def test_d1_old_position_removed():
    """La vecchia D1 (1.5, 0.0, π) NON deve essere più presente."""
    for x, y, yaw in SPAWN_LISTS[2]:
        is_old_d1 = (
            x == 1.5 and
            y == 0.0 and
            abs(yaw - math.pi) < 0.01
        )
        assert not is_old_d1, f"Vecchia D1 (1.5, 0.0, π) ancora presente: {(x, y, yaw)}"


def test_dr_noise_std_constant_exists():
    """Round 1 introduce DR_NOISE_STD = 0.02 (gaussian noise LIDAR training-only)."""
    assert hasattr(_env, 'DR_NOISE_STD'), "DR_NOISE_STD constant mancante in usv_env"
    assert abs(_env.DR_NOISE_STD - 0.02) < 1e-9, f"DR_NOISE_STD atteso 0.02, got {_env.DR_NOISE_STD}"


def test_step_action_accepts_training_kwarg():
    """step_action deve accettare keyword 'training: bool = True' per DR."""
    import inspect
    UsvEnv = _env.UsvEnv
    sig = inspect.signature(UsvEnv.step_action)
    assert 'training' in sig.parameters, "step_action manca parametro 'training'"
    param = sig.parameters['training']
    assert param.default is True, f"training default atteso True, got {param.default}"
    # Verifica che il tipo sia bool tramite annotation
    assert param.annotation is bool, f"training annotation attesa bool, got {param.annotation}"


def test_push_frame_uses_clean_scan_when_no_arg():
    """_push_frame() senza arg deve usare self.current_scan (no noise)."""
    import numpy as np
    import inspect
    UsvEnv = _env.UsvEnv
    sig = inspect.signature(UsvEnv._push_frame)
    # Verifica signature: deve avere parametro 'scan' con default None
    assert 'scan' in sig.parameters, "_push_frame deve accettare param 'scan'"
    assert sig.parameters['scan'].default is None, "_push_frame scan default atteso None"


def test_push_frame_uses_provided_scan_when_passed():
    """_push_frame(custom_scan) deve usare lo scan custom, non self.current_scan.

    Garanzia chiave: in training, step_action passa scan rumoroso a _push_frame
    SENZA mutare self.current_scan. Questo elimina rischio temporal coupling.
    """
    import numpy as np
    from collections import deque
    UsvEnv = _env.UsvEnv

    # Costruisci un instance senza __init__ (evita ROS init)
    env = UsvEnv.__new__(UsvEnv)
    env.current_scan = np.ones(_env.LIDAR_BEAMS, dtype=np.float32) * 3.0  # clean: 3m
    env._frame_buffer = deque(maxlen=_env.FRAME_STACK)

    # Scan custom diverso da current_scan
    custom = np.ones(_env.LIDAR_BEAMS, dtype=np.float32) * 1.5  # noisy: 1.5m
    env._push_frame(custom)

    # Il buffer deve contenere il custom scan normalizzato (1.5/5.0 = 0.3)
    last_frame = env._frame_buffer[-1]
    assert abs(last_frame[0] - 0.3) < 1e-6, (
        f"_push_frame ha usato self.current_scan invece del custom scan. "
        f"Atteso 0.3 (1.5/5.0), got {last_frame[0]}"
    )
    # self.current_scan NON deve essere stato modificato
    assert abs(env.current_scan[0] - 3.0) < 1e-6, (
        f"_push_frame ha mutato self.current_scan: atteso 3.0, got {env.current_scan[0]}"
    )

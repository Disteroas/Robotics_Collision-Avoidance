import sys, os
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
    import math
    for entry in SPAWN_LISTS[2]:
        yaw = entry[2]
        assert 0.0 <= yaw < 2 * math.pi + 0.01, f"Yaw {yaw} fuori range [0, 2π)"


def test_spawn_safety_dist_is_0_40():
    assert _env.SPAWN_SAFETY_DIST == 0.40


def test_spawn_max_retries_is_3():
    assert _env.SPAWN_MAX_RETRIES == 3

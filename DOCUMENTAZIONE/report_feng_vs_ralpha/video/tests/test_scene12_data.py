import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scene12_data as sd


def test_m3_success_by_seed_counts():
    feng = dict(sd.m3_success_by_seed("Feng"))
    ral = dict(sd.m3_success_by_seed("r_alpha"))
    assert len(feng) == 10 and len(ral) == 10
    # report: 3/10 vs 7/10 seeds generalize (success_rate > 0)
    assert sum(1 for v in feng.values() if v > 0) == 3
    assert sum(1 for v in ral.values() if v > 0) == 7
    assert all(0.0 <= v <= 1.0 for v in list(feng.values()) + list(ral.values()))


def test_m2_success_by_seed():
    feng = dict(sd.m2_success_by_seed("Feng"))
    assert len(feng) == 10
    assert all(0.0 <= v <= 1.0 for v in feng.values())


def test_crash_episode_actions():
    # kinematic pocket spawn must exist for feng seed 0 on M2
    ep = sd.crash_episode("Feng", 0, "(-7.0,5.0)", maze=2)
    assert ep["spawn"] == "(-7.0,5.0)"
    assert len(ep["actions"]) > 0
    assert all(0 <= a <= 10 for a in ep["actions"])
    assert ep["actions"] == ep["actions"]  # ordered by step
    # perceptual spawn too
    ep2 = sd.crash_episode("Feng", 0, "(-6.0,0.0)", maze=2)
    assert len(ep2["actions"]) > 0

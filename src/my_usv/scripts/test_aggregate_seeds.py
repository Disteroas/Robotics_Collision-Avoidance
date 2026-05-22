import csv
import numpy as np
from aggregate_seeds import iqm, bootstrap_ci, aggregate, read_summaries


def test_iqm_small_sample_is_mean():
    assert abs(iqm([0.4, 0.5, 0.6]) - 0.5) < 1e-9


def test_iqm_trims_quartiles():
    # n=8: trim i 2 estremi per lato → media di [3,4,5,6] = 4.5
    assert abs(iqm([1, 2, 3, 4, 5, 6, 7, 8]) - 4.5) < 1e-9


def test_bootstrap_ci_constant_values():
    lo, hi = bootstrap_ci([0.5, 0.5, 0.5], n_resamples=200, seed=0)
    assert abs(lo - 0.5) < 1e-9 and abs(hi - 0.5) < 1e-9


def test_aggregate_groups_by_maze(tmp_path):
    def write(p, rows):
        with open(p, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=[
                'config', 'seed', 'maze', 'episodes',
                'n_success', 'success_rate', 'avg_reward', 'avg_steps'])
            w.writeheader()
            for r in rows:
                w.writerow(r)

    base = {'config': 'x', 'episodes': 10, 'n_success': 4,
            'avg_reward': 100, 'avg_steps': 200}
    p0 = tmp_path / 's0.csv'
    p1 = tmp_path / 's1.csv'
    write(p0, [{**base, 'seed': 0, 'maze': 1, 'success_rate': 0.4}])
    write(p1, [{**base, 'seed': 1, 'maze': 1, 'success_rate': 0.6}])

    rows = read_summaries([str(p0), str(p1)])
    out = aggregate(rows)
    assert len(out) == 1
    r = out[0]
    assert r['maze'] == 1 and r['n_seed'] == 2
    assert abs(r['mean'] - 0.5) < 1e-9


def test_aggregate_single_seed_std_is_nan():
    import math
    rows = [{'config': 'x', 'seed': 0, 'maze': 1, 'episodes': 10,
             'n_success': 7, 'success_rate': 0.7, 'avg_reward': 100,
             'avg_steps': 200}]
    out = aggregate(rows)
    assert out[0]['n_seed'] == 1
    assert math.isnan(out[0]['std'])

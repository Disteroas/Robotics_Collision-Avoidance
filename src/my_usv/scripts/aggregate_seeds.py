"""Aggrega gli eval_summary.csv di N seed in statistiche robuste.

Reporting (Henderson 2018, Agarwal 2021): mai il max. Sempre mean±std,
IQM (Inter-Quartile Mean) e intervallo di confidenza 95% via bootstrap.
Solo numpy + stdlib (niente pandas: non garantito nel container).
"""
import argparse
import csv
import glob
import os

import numpy as np


def iqm(values) -> float:
    v = np.sort(np.asarray(values, dtype=float))
    n = len(v)
    if n == 0:
        return float('nan')
    lo = int(np.floor(n * 0.25))
    hi = int(np.ceil(n * 0.75))
    trimmed = v[lo:hi] if hi > lo else v
    return float(np.mean(trimmed))


def bootstrap_ci(values, n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 0):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    means = np.array([
        np.mean(rng.choice(v, size=len(v), replace=True))
        for _ in range(n_resamples)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (lo, hi)


def read_summaries(paths):
    rows = []
    for p in paths:
        with open(p, newline='') as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def aggregate(rows):
    by_maze = {}
    for r in rows:
        by_maze.setdefault(int(r['maze']), []).append(float(r['success_rate']))
    out = []
    for maze in sorted(by_maze):
        sr = by_maze[maze]
        lo, hi = bootstrap_ci(sr)
        out.append({
            'maze':    maze,
            'n_seed':  len(sr),
            'mean':    float(np.mean(sr)),
            'std':     float(np.std(sr, ddof=1)) if len(sr) > 1 else float('nan'),
            'iqm':     iqm(sr),
            'ci_low':  lo,
            'ci_high': hi,
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, default='default')
    p.add_argument('--runs-dir', type=str, default='runs')
    p.add_argument('--output', type=str, required=True)
    args = p.parse_args()

    pattern = os.path.join(args.runs_dir, args.config, 'seed_*', 'eval_summary.csv')
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[ERRORE] Nessun eval_summary.csv in {pattern}")
        raise SystemExit(1)

    rows = read_summaries(paths)
    out = aggregate(rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'maze', 'n_seed', 'mean', 'std', 'iqm', 'ci_low', 'ci_high'])
        w.writeheader()
        for r in out:
            w.writerow(r)

    print(f"  Aggregato {len(paths)} seed -> {args.output}")
    for r in out:
        print(f"  M{r['maze']}: {r['mean']*100:.1f}% +/- {r['std']*100:.1f}  "
              f"(IQM {r['iqm']*100:.1f}%; 95% CI [{r['ci_low']*100:.1f}, "
              f"{r['ci_high']*100:.1f}]) n={r['n_seed']}")


if __name__ == '__main__':
    main()

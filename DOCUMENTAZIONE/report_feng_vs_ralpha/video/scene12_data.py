"""Data layer for scena-12 result animations. Reads the real N=10 eval CSVs
from runs/ via style.seed_dir. No matplotlib here — pure pandas, unit-tested."""
import os

import pandas as pd

import style  # reuse REGISTRY / seed_dir / ROOT

SEEDS = list(range(10))


def _summary(config, seed):
    path = os.path.join(style.seed_dir(config, seed), "eval_summary.csv")
    return pd.read_csv(path)


def _success_by_seed(config, maze):
    out = []
    for s in SEEDS:
        df = _summary(config, s)
        row = df[df["maze"] == maze]
        if not row.empty:
            out.append((s, float(row["success_rate"].iloc[0])))
    return out


def m3_success_by_seed(config):
    """[(seed, success_rate)] for the unseen maze M3."""
    return _success_by_seed(config, 3)


def m2_success_by_seed(config):
    """[(seed, success_rate)] for the training maze M2."""
    return _success_by_seed(config, 2)


def maze_mean(config, maze):
    """Mean success_rate (%) over seeds for one maze — for the results table."""
    vals = [v for _, v in _success_by_seed(config, maze)]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0


def crash_episode(config, seed, spawn, maze=2):
    """First episode at `spawn` on `maze`; returns dict with the ordered action
    sequence and per-step front/left/right distances (for the action strip)."""
    df = style.load_eval_steps(config, seed, maze)
    df = df[df["spawn"] == spawn]
    if df.empty:
        raise ValueError(f"no steps at spawn {spawn} for {config} seed {seed} m{maze}")
    ep = int(df["episode"].iloc[0])
    ep_df = df[df["episode"] == ep].sort_values("step")
    return dict(
        spawn=spawn, episode=ep,
        actions=[int(a) for a in ep_df["action"].tolist()],
        front=ep_df["front_dist"].tolist(),
        left=ep_df["left_dist"].tolist(),
        right=ep_df["right_dist"].tolist(),
    )

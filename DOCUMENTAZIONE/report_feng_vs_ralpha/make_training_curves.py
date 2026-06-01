"""Training-reward curves (mean +/- std over 10 seeds), Feng vs r_alpha.

Usai requested this figure (MountainCar-style band). Reward scales differ
(sparse +5/-1000 for Feng, shaped for r_alpha) so the two agents are on
separate panels with separate y-axes. Curve = 100-episode moving average
(avg100 column of training_log.csv). Output: latex/figures/fig_training_curves.png
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "latex", "figures")
SEEDS = list(range(10))
BASE = {
    "Feng":    os.path.join(ROOT, "runs", "feng_hw_A", "seed_{s}"),
    "r_alpha": os.path.join(ROOT, "runs", "r_alpha_hw_A", "seed_{s}"),
}
COLOR = {"Feng": "#d62728", "r_alpha": "#1f77b4"}


def load_avg100(cfg, s):
    ep, val = [], []
    with open(os.path.join(BASE[cfg].format(s=s), "training_log.csv"), newline="") as f:
        for r in csv.DictReader(f):
            try:
                ep.append(int(r["ep_global"]))
                val.append(float(r["avg100"]))
            except (ValueError, KeyError):
                pass
    return np.array(ep), np.array(val)


def stack(cfg):
    curves = [load_avg100(cfg, s)[1] for s in SEEDS]
    n = min(len(c) for c in curves)
    M = np.array([c[:n] for c in curves])      # (n_seed, n_ep)
    x = np.arange(1, n + 1)
    return x, M


fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
for ax, cfg in zip(axes, ("Feng", "r_alpha")):
    x, M = stack(cfg)
    c = COLOR[cfg]
    for row in M:                              # one faint line per seed
        ax.plot(x, row, color=c, lw=0.6, alpha=0.28, zorder=2)
    ax.plot(x, M.mean(0), color="black", lw=1.8, zorder=4,
            label="mean of 10 seeds")
    ax.set_title(cfg, fontsize=12, fontweight="bold")
    ax.set_xlabel("training episode", fontsize=10)
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=9)
    ax.legend(fontsize=8, loc="upper left")
axes[0].set_ylabel("reward (100-ep avg)", fontsize=10)
fig.tight_layout()
out = os.path.join(OUT, "fig_training_curves.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved:", out)

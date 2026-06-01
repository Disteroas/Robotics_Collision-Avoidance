"""Figure restanti per il report feng-vs-r_alpha.

fig_scatter_compare : success per-seed feng vs r_alpha (Machine A), per maze.
fig_cross_hardware  : r_alpha Machine A vs B, per maze, mean + 95% CI + seed dots.
fig_crash_taxonomy  : crash front scomposti in kinematic / perceptual / other.

Solo numpy + matplotlib + stdlib.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "latex", "figures")
os.makedirs(OUT, exist_ok=True)

SEEDS_A = list(range(10))   # feng_hw_A, r_alpha_hw_A: N=10
SEEDS_B = [0, 1, 2, 3, 4]   # r_alpha_hw_B: 5 seed (cross-HW = pairing same-seed)
MAZES = [1, 2, 3]
EXTREME = {0, 1, 9, 10}
CENTRAL = {4, 5, 6}
BASE = {
    "feng":      os.path.join(ROOT, "runs", "feng_hw_A", "seed_{s}"),
    "r_alpha":   os.path.join(ROOT, "runs", "r_alpha_hw_A", "seed_{s}"),
    "r_alpha_B": os.path.join(ROOT, "runs", "r_alpha_hw_B", "r_alpha", "seed_{s}"),
}


def seeds_for(cfg):
    return SEEDS_B if cfg == "r_alpha_B" else SEEDS_A
plt.rcParams.update({"font.size": 13, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})
rng = np.random.default_rng(42)


def sdir(cfg, s):
    return BASE[cfg].format(s=s)


def read_summary(cfg, s):
    out = {}
    with open(os.path.join(sdir(cfg, s), "eval_summary.csv"), newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["maze"])] = float(r["success_rate"]) * 100
    return out


def bootstrap_ci(v, n=10000, alpha=0.05):
    v = np.asarray(v, float)
    means = np.array([np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)])
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


S = {cfg: {s: read_summary(cfg, s) for s in seeds_for(cfg)}
     for cfg in ("feng", "r_alpha", "r_alpha_B")}

# ============================================================ FIG scatter compare
fig, ax = plt.subplots(figsize=(7.6, 4.8))
configs = [("feng", "Feng", "#d62728"), ("r_alpha", "r_alpha", "#1f77b4")]
width = 0.34
for ci, (cfg, lab, col) in enumerate(configs):
    for mi, m in enumerate(MAZES):
        vals = np.array([S[cfg][s][m] for s in SEEDS_A])
        xc = mi + (ci - 0.5) * width
        lo, hi = bootstrap_ci(vals)
        mean = vals.mean()
        ax.errorbar(xc, mean, yerr=[[mean - lo], [hi - mean]], fmt="_",
                    color="black", capsize=6, lw=1.8, zorder=3)
        jit = (rng.random(len(SEEDS_A)) - 0.5) * 0.18
        ax.scatter(xc + jit, vals, color=col, s=55, edgecolor="k", linewidth=0.5,
                   zorder=4, alpha=0.9,
                   label=lab if mi == 0 else None)
ax.set_xticks(range(len(MAZES)))
ax.set_xticklabels([f"Maze {m}" for m in MAZES])
ax.set_ylabel("Success rate (%)")
ax.set_ylim(-5, 105)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_scatter_compare.png"), dpi=150)
plt.close(fig)

# ============================================================ FIG cross-hardware
fig, ax = plt.subplots(figsize=(7.6, 4.8))
hw = [("r_alpha", "Machine A", "#1f77b4"), ("r_alpha_B", "Machine B", "#ff7f0e")]
for ci, (cfg, lab, col) in enumerate(hw):
    for mi, m in enumerate(MAZES):
        vals = np.array([S[cfg][s][m] for s in SEEDS_B])  # paired same-seed 0-4
        xc = mi + (ci - 0.5) * width
        lo, hi = bootstrap_ci(vals)
        mean = vals.mean()
        ax.errorbar(xc, mean, yerr=[[mean - lo], [hi - mean]], fmt="_",
                    color="black", capsize=6, lw=1.8, zorder=3)
        jit = (rng.random(len(SEEDS_B)) - 0.5) * 0.18
        ax.scatter(xc + jit, vals, color=col, s=55, edgecolor="k", linewidth=0.5,
                   zorder=4, alpha=0.9, label=lab if mi == 0 else None)
ax.set_xticks(range(len(MAZES)))
ax.set_xticklabels([f"Maze {m}" for m in MAZES])
ax.set_ylabel("Success rate (%)")
ax.set_ylim(-5, 105)
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_cross_hardware.png"), dpi=150)
plt.close(fig)

# ============================================================ FIG crash taxonomy
def crash_rows(cfg, s, m):
    p = os.path.join(sdir(cfg, s), f"eval_crashes_m{m}.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def last_action(row):
    la = (row.get("last_actions") or "").strip().strip('"')
    try:
        return int(la.split(",")[-1])
    except (ValueError, IndexError):
        return None


cats = ["kinematic\n(front+max turn)", "perceptual\n(front, not max turn)", "side\n(left/right)"]
data = {}
for cfg in ("feng", "r_alpha"):
    kin = perc = side = tot = 0
    for s in SEEDS_A:
        for m in MAZES:
            for r in crash_rows(cfg, s, m):
                tot += 1
                sec = (r.get("crash_sector") or "").strip()
                a = last_action(r)
                if sec == "front":
                    if a in EXTREME:
                        kin += 1
                    else:                      # front but not a near-maximal turn
                        perc += 1
                elif sec in ("left", "right"):
                    side += 1
    data[cfg] = [100 * kin / tot, 100 * perc / tot, 100 * side / tot]  # % of crashes

fig, ax = plt.subplots(figsize=(7.0, 4.6))
x = np.arange(len(cats))
w = 0.38
ax.bar(x - w / 2, data["feng"], w, label="Feng", color="#d62728", edgecolor="k", linewidth=0.5)
ax.bar(x + w / 2, data["r_alpha"], w, label="r_alpha", color="#1f77b4", edgecolor="k", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.set_ylabel("Share of each agent's crashes (%)")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_crash_taxonomy.png"), dpi=150)
plt.close(fig)

print("Scritte:")
for f in sorted(os.listdir(OUT)):
    print("  ", f)
print("\ncrash taxonomy counts:", data)

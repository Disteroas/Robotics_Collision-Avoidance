"""Genera le figure PNG per il report 5-seed (Macchina A).

Sorgenti: runs/r_alpha/seed_{0..4}/{eval_summary.csv, training_log.csv, eval_crashes_m*.csv}
Output:   DOCUMENTAZIONE/report_5seed_riproducibilita/figures/*.png

HW anonimizzato (Macchina A). Solo numpy + matplotlib + stdlib.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS = os.path.join(ROOT, "runs", "r_alpha")
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

SEEDS = [0, 1, 2, 3, 4]
MAZES = [1, 2, 3]
COLORS = {0: "#1f77b4", 1: "#2ca02c", 2: "#d62728", 3: "#9467bd", 4: "#ff7f0e"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def read_summary(seed):
    """maze -> dict(success_rate, avg_reward, avg_steps)"""
    out = {}
    with open(os.path.join(RUNS, f"seed_{seed}", "eval_summary.csv"), newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["maze"])] = {
                "sr": float(r["success_rate"]),
                "rew": float(r["avg_reward"]),
                "steps": float(r["avg_steps"]),
            }
    return out


def read_training(seed):
    ep, avg100 = [], []
    with open(os.path.join(RUNS, f"seed_{seed}", "training_log.csv"), newline="") as f:
        for r in csv.DictReader(f):
            ep.append(int(r["ep_global"]))
            try:
                avg100.append(float(r["avg100"]))
            except (ValueError, KeyError):
                avg100.append(np.nan)
    return np.array(ep), np.array(avg100)


def read_crash_sectors(seed, maze):
    sectors = []
    p = os.path.join(RUNS, f"seed_{seed}", f"eval_crashes_m{maze}.csv")
    if not os.path.exists(p):
        return sectors
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            s = (r.get("crash_sector") or "").strip()
            if s:
                sectors.append(s)
    return sectors


def bootstrap_ci(values, n=10000, alpha=0.05, seed=0):
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(rng.choice(v, len(v), replace=True)) for _ in range(n)])
    return np.percentile(means, 100 * alpha / 2), np.percentile(means, 100 * (1 - alpha / 2))


summ = {s: read_summary(s) for s in SEEDS}

# ---------------------------------------------------------------- FIG 1
# Success-rate per-seed (punti) + mean + 95% CI per maze
fig, ax = plt.subplots(figsize=(7.5, 4.8))
rng = np.random.default_rng(42)
for mi, m in enumerate(MAZES):
    vals = np.array([summ[s][m]["sr"] * 100 for s in SEEDS])
    mean = vals.mean()
    lo, hi = bootstrap_ci(vals / 100)
    lo, hi = lo * 100, hi * 100
    # CI bar
    ax.errorbar(mi, mean, yerr=[[mean - lo], [hi - mean]], fmt="_", color="black",
                capsize=8, lw=2, zorder=3, label="mean + 95% CI" if mi == 0 else None)
    ax.scatter([mi] * 100, [mean] * 100, alpha=0)  # spacing
    # seed dots (jitter)
    jit = (rng.random(len(SEEDS)) - 0.5) * 0.22
    for j, s in enumerate(SEEDS):
        ax.scatter(mi + jit[j], vals[j], color=COLORS[s], s=70, edgecolor="k",
                   linewidth=0.6, zorder=4,
                   label=f"seed {s}" if mi == 0 else None)
ax.set_xticks(range(len(MAZES)))
ax.set_xticklabels([f"Maze {m}" for m in MAZES])
ax.set_ylabel("Success rate (%)")
ax.set_ylim(-5, 105)
ax.set_title("Success rate per seed — Macchina A, n=5 (eval round-robin, $\\epsilon$=0)")
ax.legend(loc="lower right", fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_success_per_seed.png"))
plt.close(fig)

# ---------------------------------------------------------------- FIG 2
# Curve training avg100 vs episodio
fig, ax = plt.subplots(figsize=(8.5, 4.8))
for s in SEEDS:
    ep, a = read_training(s)
    lw = 2.4 if s == 2 else 1.3
    alpha = 0.95 if s == 2 else 0.8
    ax.plot(ep, a, color=COLORS[s], lw=lw, alpha=alpha,
            label=f"seed {s}" + (" (collasso)" if s == 2 else ""))
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Episodio globale")
ax.set_ylabel("Reward medio (avg100)")
ax.set_title("Curve di training — Macchina A, 5 seed, 5000 episodi")
ax.legend(loc="upper left", fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_training_curves.png"))
plt.close(fig)

# ---------------------------------------------------------------- FIG 3
# Crash-sector failure mode: conteggi aggregati su tutti i seed, per maze
sector_names = ["front", "left", "right"]
counts = {m: {sec: 0 for sec in sector_names} for m in MAZES}
for s in SEEDS:
    for m in MAZES:
        for sec in read_crash_sectors(s, m):
            if sec in counts[m]:
                counts[m][sec] += 1
            else:
                counts[m].setdefault(sec, 0)
                counts[m][sec] += 1
all_sectors = sorted({sec for m in MAZES for sec in counts[m]})
fig, ax = plt.subplots(figsize=(7.5, 4.8))
x = np.arange(len(MAZES))
w = 0.8 / max(1, len(all_sectors))
sec_colors = {"front": "#d62728", "left": "#1f77b4", "right": "#ff7f0e"}
for k, sec in enumerate(all_sectors):
    vals = [counts[m].get(sec, 0) for m in MAZES]
    ax.bar(x + k * w, vals, w, label=sec, color=sec_colors.get(sec, None),
           edgecolor="k", linewidth=0.5)
ax.set_xticks(x + w * (len(all_sectors) - 1) / 2)
ax.set_xticklabels([f"Maze {m}" for m in MAZES])
ax.set_ylabel("N. crash (somma 5 seed)")
ax.set_title("Failure mode: settore di collisione per maze")
ax.legend(title="settore", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_crash_sectors.png"))
plt.close(fig)

# ---------------------------------------------------------------- FIG 4
# Boxplot reward e steps medi per maze (campioni = 5 seed)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.4))
rew_data = [[summ[s][m]["rew"] for s in SEEDS] for m in MAZES]
step_data = [[summ[s][m]["steps"] for s in SEEDS] for m in MAZES]
for ax, data, title, ylab in [
    (axes[0], rew_data, "Reward medio per maze", "avg_reward"),
    (axes[1], step_data, "Step sopravvissuti per maze", "avg_steps"),
]:
    bp = ax.boxplot(data, tick_labels=[f"M{m}" for m in MAZES], patch_artist=True,
                    widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#cfe2f3")
    for mi, m in enumerate(MAZES):
        ys = data[mi]
        ax.scatter([mi + 1] * len(ys), ys, color=[COLORS[s] for s in SEEDS],
                   edgecolor="k", linewidth=0.5, s=45, zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylab)
axes[1].axhline(500, color="gray", lw=0.8, ls="--")
axes[1].text(0.5, 505, "limite 500 step", fontsize=8, color="gray")
fig.suptitle("Distribuzione sui 5 seed — Macchina A", y=1.00)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_reward_steps_box.png"), bbox_inches="tight")
plt.close(fig)

print("Figure scritte in", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)

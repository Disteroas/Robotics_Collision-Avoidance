"""Analisi estese per il report feng-vs-r_alpha (Macchina A) + cross-hardware (B).

Metriche (oltre alla tabella success già prodotta da aggregate_seeds.py):
  1. Performance profile (frazione run con success >= tau) feng vs r_alpha @A, con bande bootstrap.
  2. Probability of improvement P(r_alpha > feng) per maze + aggregata (rliable, Agarwal 2021).
  3. Spearman rank corr fra reward finale di training (avg100) ed eval success (r_alpha @A).
  4. Tassonomia crash: front+sterzo-estremo = CINEMATICO (Classe A) vs front+centrale = PERCETTIVO (Classe B).
  5. q_spread / q_chosen medi (eval_steps) -> indifferenza della policy.

Solo numpy + matplotlib + stdlib. Host Python.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(OUT, exist_ok=True)

SEEDS_A = list(range(10))   # feng_hw_A, r_alpha_hw_A: N=10
SEEDS_B = [0, 1, 2, 3, 4]   # r_alpha_hw_B: 5 seed (cross-HW = pairing same-seed)
MAZES = [1, 2, 3]


def seeds_for(cfg):
    return SEEDS_B if cfg == "r_alpha_B" else SEEDS_A

# basi path (rB ha sottocartella r_alpha/)
BASE = {
    "feng":     os.path.join(ROOT, "runs", "feng_hw_A", "seed_{s}"),
    "r_alpha":  os.path.join(ROOT, "runs", "r_alpha_hw_A", "seed_{s}"),
    "r_alpha_B": os.path.join(ROOT, "runs", "r_alpha_hw_B", "r_alpha", "seed_{s}"),
}
EXTREME = {0, 1, 9, 10}   # sterzo quasi-massimo (azioni 0..10, 5=dritto)
CENTRAL = {4, 5, 6}       # quasi-dritto

rng = np.random.default_rng(0)


def seed_dir(cfg, s):
    return BASE[cfg].format(s=s)


def read_summary(cfg, s):
    out = {}
    with open(os.path.join(seed_dir(cfg, s), "eval_summary.csv"), newline="") as f:
        for r in csv.DictReader(f):
            out[int(r["maze"])] = float(r["success_rate"])
    return out


def final_avg100(cfg, s):
    last = None
    with open(os.path.join(seed_dir(cfg, s), "training_log.csv"), newline="") as f:
        for r in csv.DictReader(f):
            try:
                last = float(r["avg100"])
            except (ValueError, KeyError):
                pass
    return last


def crash_rows(cfg, s, maze):
    p = os.path.join(seed_dir(cfg, s), f"eval_crashes_m{maze}.csv")
    if not os.path.exists(p):
        return []
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def last_action(row):
    la = (row.get("last_actions") or "").strip().strip('"')
    if not la:
        return None
    try:
        return int(la.split(",")[-1])
    except ValueError:
        return None


def q_stats(cfg, s, maze):
    """media q_spread e q_chosen sul file eval_steps (può essere grande)."""
    p = os.path.join(seed_dir(cfg, s), f"eval_steps_m{maze}.csv")
    if not os.path.exists(p):
        return (np.nan, np.nan)
    spreads, chosen = [], []
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            try:
                spreads.append(float(r["q_spread"]))
                chosen.append(float(r["q_chosen"]))
            except (ValueError, KeyError):
                pass
    if not spreads:
        return (np.nan, np.nan)
    return (float(np.mean(spreads)), float(np.mean(chosen)))


# ---------- carica success per (cfg, seed, maze) ----------
S = {cfg: {s: read_summary(cfg, s) for s in seeds_for(cfg)}
     for cfg in ("feng", "r_alpha", "r_alpha_B")}

report = {}

# ============================================================ 1. PERF PROFILE
taus = np.linspace(0, 1, 101)


def profile(scores, taus):
    scores = np.asarray(scores, float)
    return np.array([(scores >= t).mean() for t in taus])


def profile_with_ci(scores, taus, n=10000):
    scores = np.asarray(scores, float)
    base = profile(scores, taus)
    boots = np.empty((n, len(taus)))
    for i in range(n):
        samp = rng.choice(scores, len(scores), replace=True)
        boots[i] = profile(samp, taus)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return base, lo, hi


feng_scores = [S["feng"][s][m] for s in SEEDS_A for m in MAZES]   # 30 run (N=10)
ra_scores = [S["r_alpha"][s][m] for s in SEEDS_A for m in MAZES]
fb, flo, fhi = profile_with_ci(feng_scores, taus)
rb, rlo, rhi = profile_with_ci(ra_scores, taus)

plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.plot(taus, fb, color="#d62728", lw=2, label="Feng")
ax.fill_between(taus, flo, fhi, color="#d62728", alpha=0.18)
ax.plot(taus, rb, color="#1f77b4", lw=2, label="r_alpha")
ax.fill_between(taus, rlo, rhi, color="#1f77b4", alpha=0.18)
ax.set_xlabel(r"Success-rate threshold $\tau$")
ax.set_ylabel(r"Fraction of runs with success $\geq \tau$")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_perf_profile.png"), dpi=150)
plt.close(fig)

# ============================================================ 2. PROB OF IMPROVEMENT
def prob_improve(x, y):
    """P(X>Y)+0.5 P(X=Y) su tutte le coppie."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    gt = sum(xi > yj for xi in x for yj in y)
    eq = sum(xi == yj for xi in x for yj in y)
    return (gt + 0.5 * eq) / (len(x) * len(y))


poi = {}
for m in MAZES:
    x = [S["r_alpha"][s][m] for s in SEEDS_A]
    y = [S["feng"][s][m] for s in SEEDS_A]
    poi[m] = prob_improve(x, y)
poi_overall = float(np.mean(list(poi.values())))
report["prob_improvement_r_alpha_over_feng"] = {
    "per_maze": {f"M{m}": round(poi[m], 3) for m in MAZES},
    "aggregate": round(poi_overall, 3),
}

# ============================================================ 3. SPEARMAN train vs eval
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


train_reward = {s: final_avg100("r_alpha", s) for s in SEEDS_A}
report["r_alpha_A_final_avg100"] = {f"s{s}": round(train_reward[s], 1) for s in SEEDS_A}
sp = {}
tr_vec = [train_reward[s] for s in SEEDS_A]
for m in MAZES:
    ev = [S["r_alpha"][s][m] for s in SEEDS_A]
    sp[f"M{m}"] = round(spearman(tr_vec, ev), 3)
mean_eval = [np.mean([S["r_alpha"][s][m] for m in MAZES]) for s in SEEDS_A]
sp["mean_eval"] = round(spearman(tr_vec, mean_eval), 3)
report["spearman_trainreward_vs_evalsuccess"] = sp

# ============================================================ 4. TASSONOMIA CRASH
crash_tax = {}
for cfg in ("feng", "r_alpha", "r_alpha_B"):
    tot = kin = perc = other = front = 0
    for s in seeds_for(cfg):
        for m in MAZES:
            for r in crash_rows(cfg, s, m):
                tot += 1
                sec = (r.get("crash_sector") or "").strip()
                a = last_action(r)
                if sec == "front":
                    front += 1
                    if a in EXTREME:
                        kin += 1
                    elif a in CENTRAL:
                        perc += 1
                    else:
                        other += 1
    crash_tax[cfg] = {
        "n_crash": tot, "front": front,
        "kinematic_front_extreme": kin,
        "perceptual_front_central": perc,
        "front_other_action": other,
        "kin_share_of_front": round(kin / front, 3) if front else None,
        "kin_share_of_all": round(kin / tot, 3) if tot else None,
    }
report["crash_taxonomy"] = crash_tax

# ============================================================ 5. Q DIAGNOSTICS
qd = {}
for cfg in ("feng", "r_alpha"):
    spreads, chosens = [], []
    for s in seeds_for(cfg):
        for m in MAZES:
            sp_, ch_ = q_stats(cfg, s, m)
            if not np.isnan(sp_):
                spreads.append(sp_); chosens.append(ch_)
    qd[cfg] = {
        "mean_q_spread": round(float(np.mean(spreads)), 3),
        "mean_q_chosen": round(float(np.mean(chosens)), 3),
        "spread_over_chosen_pct": round(100 * np.mean(spreads) / np.mean(chosens), 2),
    }
report["q_diagnostics"] = qd

print(json.dumps(report, indent=2))
print("\nFigure: fig_perf_profile.png in", OUT)

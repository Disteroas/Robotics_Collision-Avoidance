"""
Analisi training feng_direct — 3000 ep, Maze 2, BETA_DECAY=0.999
Genera 5 plot salvati in plots_feng_direct/
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

OUT_DIR = Path("script_Davide_python/plots_feng_direct")
OUT_DIR.mkdir(exist_ok=True)

TRAIN_CSV = "src/my_usv/scripts/training_log.csv"
TEST_CSV  = "src/my_usv/scripts/test_results.csv"

train = pd.read_csv(TRAIN_CSV)
test  = pd.read_csv(TEST_CSV)

# ── helpers ──────────────────────────────────────────────────────────────────
def rolling_crash(series, w=100):
    return series.rolling(w, min_periods=1).mean() * 100

def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved: {p}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 1 — Training curve: reward/ep + avg100
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(train.ep_global, train.reward, color="#aaaaaa", lw=0.4, alpha=0.6,
        label="Reward per ep")
ax.plot(train.ep_global, train.avg100, color="#e74c3c", lw=1.8,
        label="Avg-100 reward")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.axhline(2500, color="#2ecc71", lw=0.8, ls="--", label="Reward max (500 step)")

ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Reward", fontsize=11)
ax.set_title("Training — reward per episode + avg-100 (Maze 2, 3000 ep)", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
save(fig, "01_training_reward.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 2 — Rolling crash rate (100-ep window)
# ═══════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 4))
crash_roll = rolling_crash(train.crashed)
ax.plot(train.ep_global, crash_roll, color="#e67e22", lw=1.8)
ax.axhline(100, color="#c0392b", lw=0.8, ls="--", label="100%")
ax.axhline(50,  color="#f39c12", lw=0.8, ls="--", label="50%")

ax.fill_between(train.ep_global, crash_roll, 100, alpha=0.08, color="#e67e22")
ax.set_ylim(0, 105)
ax.set_xlabel("Episode", fontsize=11)
ax.set_ylabel("Crash rate (%, 100-ep window)", fontsize=11)
ax.set_title("Training — rolling crash rate", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
save(fig, "02_training_crash_rate.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 3 — Epsilon + avg_loss durante training
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

ax1.plot(train.ep_global, train.epsilon, color="#3498db", lw=1.5)
ax1.set_ylabel("ε (epsilon)", fontsize=10)
ax1.set_title("Training — epsilon decay + loss", fontsize=13)
ax1.grid(True, alpha=0.3)

loss_roll = train.avg_loss.rolling(50, min_periods=1).mean()
ax2.plot(train.ep_global, train.avg_loss, color="#cccccc", lw=0.4, alpha=0.5)
ax2.plot(train.ep_global, loss_roll, color="#9b59b6", lw=1.5, label="Avg-50 loss")
ax2.set_ylabel("Avg loss", fontsize=10)
ax2.set_xlabel("Episode", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
save(fig, "03_training_epsilon_loss.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 4 — Test: crash rate + reward boxplot per maze
# ═══════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

maze_labels = ["Maze 1\n(training, unseen)", "Maze 2\n(training)", "Maze 3\n(test only)"]
colors      = ["#3498db", "#e74c3c", "#2ecc71"]
crash_rates = []
reward_data = []

for m in [1, 2, 3]:
    df = test[test.maze_id == m]
    crash_rates.append(df.crashed.mean() * 100)
    reward_data.append(df.reward.values)

# Bar: crash rate
bars = ax1.bar(maze_labels, crash_rates, color=colors, edgecolor="black", linewidth=0.8)
for bar, rate in zip(bars, crash_rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{rate:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylim(0, 115)
ax1.set_ylabel("Crash rate (%)", fontsize=11)
ax1.set_title("Test — crash rate per maze", fontsize=12)
ax1.grid(True, axis="y", alpha=0.3)

# Boxplot: reward
bp = ax2.boxplot(reward_data, patch_artist=True, notch=False,
                 medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax2.set_xticklabels(maze_labels)
ax2.axhline(0, color="black", lw=0.8, ls="--")
ax2.axhline(2500, color="#27ae60", lw=0.8, ls="--", label="Max reward (500 step)")
ax2.set_ylabel("Reward", fontsize=11)
ax2.set_title("Test — reward distribution per maze", fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
save(fig, "04_test_results.png")

# ═══════════════════════════════════════════════════════════════════════════
# PLOT 5 — Dashboard riepilogativa
# ═══════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# (0,0)-(0,1) — training reward
ax_r = fig.add_subplot(gs[0, :2])
ax_r.plot(train.ep_global, train.reward, color="#bbbbbb", lw=0.3, alpha=0.5)
ax_r.plot(train.ep_global, train.avg100, color="#e74c3c", lw=1.8, label="Avg-100")
ax_r.axhline(0, color="black", lw=0.7, ls="--")
ax_r.axhline(2500, color="#2ecc71", lw=0.7, ls="--")
ax_r.set_title("Training reward (Maze 2, 3000 ep)", fontsize=11)
ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Reward")
ax_r.legend(fontsize=8); ax_r.grid(True, alpha=0.3)

# (0,2) — crash rate training
ax_c = fig.add_subplot(gs[0, 2])
ax_c.plot(train.ep_global, rolling_crash(train.crashed), color="#e67e22", lw=1.5)
ax_c.set_ylim(0, 105); ax_c.set_title("Rolling crash rate (train)", fontsize=11)
ax_c.set_xlabel("Episode"); ax_c.set_ylabel("Crash % (100-ep)")
ax_c.grid(True, alpha=0.3)

# (1,0) — epsilon
ax_e = fig.add_subplot(gs[1, 0])
ax_e.plot(train.ep_global, train.epsilon, color="#3498db", lw=1.5)
ax_e.set_title("Epsilon decay", fontsize=11)
ax_e.set_xlabel("Episode"); ax_e.set_ylabel("ε")
ax_e.grid(True, alpha=0.3)

# (1,1) — test crash rate
ax_t1 = fig.add_subplot(gs[1, 1])
bars = ax_t1.bar(["M1\n(unseen)", "M2\n(trained)", "M3\n(test)"],
                  crash_rates, color=colors, edgecolor="black", lw=0.8)
for bar, rate in zip(bars, crash_rates):
    ax_t1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f"{rate:.0f}%", ha="center", fontsize=10, fontweight="bold")
ax_t1.set_ylim(0, 115); ax_t1.set_title("Test — crash rate", fontsize=11)
ax_t1.set_ylabel("Crash %"); ax_t1.grid(True, axis="y", alpha=0.3)

# (1,2) — test reward boxplot
ax_t2 = fig.add_subplot(gs[1, 2])
bp = ax_t2.boxplot(reward_data, patch_artist=True,
                   medianprops=dict(color="black", lw=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax_t2.set_xticklabels(["M1", "M2", "M3"])
ax_t2.axhline(0, color="black", lw=0.7, ls="--")
ax_t2.set_title("Test — reward distribution", fontsize=11)
ax_t2.set_ylabel("Reward"); ax_t2.grid(True, axis="y", alpha=0.3)

fig.suptitle("DDQN USV — feng_direct  |  3000 ep Maze 2  |  BETA_DECAY=0.999",
             fontsize=14, fontweight="bold")
save(fig, "00_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════
# Stampa riepilogo testuale
# ═══════════════════════════════════════════════════════════════════════════
print("\n== RIEPILOGO ==================================================")
print(f"Training completato: {len(train)} episodi")
print(f"  Epsilon finale:   {train.epsilon.iloc[-1]:.4f}")
print(f"  Avg100 finale:    {train.avg100.iloc[-1]:.1f}")
print(f"  Crash totali:     {int(train.crashed.sum())} / {len(train)} "
      f"({train.crashed.mean()*100:.1f}%)")
print(f"  Steps totali:     {int(train.total_steps.iloc[-1]):,}")
print()
for m, label in [(1,"Maze 1 (unseen)"), (2,"Maze 2 (trained)"), (3,"Maze 3 (test)")]:
    df = test[test.maze_id == m]
    suc = (df.crashed == 0).sum()
    cr  = df.crashed.mean() * 100
    avg = df.reward.mean()
    print(f"  {label}:  crash={cr:.0f}%  successi={suc}/30  avg_reward={avg:.0f}")
print("===============================================================")

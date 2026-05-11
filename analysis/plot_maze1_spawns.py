"""
Plot Maze 1 (labirinto_9a) top-down with the 2 selected spawn points.
Wall data from <state> section of labirinto_9a.world (world-frame poses).

P1 (-2.9, -2.0, N): left channel — validated
P2 ( 1.0, -1.0, N): inner chamber — validate with test_spawns.sh
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ── Wall data: (name, cx, cy, yaw_rad, length) — world-frame from <state> ──
WALLS = [
    ("Wall_0",  -3.62886, -1.12723,  1.57080, 5.36043),
    ("Wall_1",  -3.80387,  1.49349,  3.14159, 0.50000),
    ("Wall_2",  -3.97886,  2.34777,  1.57080, 1.88957),
    ("Wall_3",  -3.80387,  3.21756,  0.00000, 0.50000),
    ("Wall_4",  -3.62886,  3.54857,  1.57080, 0.81203),
    ("Wall_5",  -2.77366,  3.86408,  0.00000, 1.86043),
    ("Wall_6",  -1.91068,  3.54857, -1.54738, 0.81221),
    ("Wall_7",   0.82634,  3.21756,  0.00000, 5.63957),
    ("Wall_8",   3.57113,  2.35553, -1.57080, 1.87406),
    ("Wall_9",   3.39614,  1.50900,  3.14159, 0.50000),
    ("Wall_10",  3.22113, -1.11947, -1.57080, 5.37594),
    ("Wall_11",  2.33584, -3.74020, -3.13283, 1.92066),
    ("Wall_12",  1.46606, -3.40918,  1.57080, 0.82754),
    ("Wall_13", -0.24090, -3.07041,  3.13251, 3.56405),
    ("Wall_15", -2.12459, -1.05744,  1.57080, 5.50000),
    ("Wall_16", -0.19959,  1.61756,  0.00000, 4.00000),
    ("Wall_17",  1.73643,  0.07197, -1.56367, 3.24126),
    ("Wall_18",  0.61464, -1.46694,  3.13958, 2.41562),
    ("Wall_19", -0.51269, -0.73215,  1.56328, 1.60622),
    ("Wall_20", -0.00162,  0.00361,  0.00646, 1.16122),
]

WALL_W = 0.15

# ── Spawn points ──
# format: (x, y, yaw, label, color, validated)
# validated=True  → min_lidar ≥ 0.40m confirmed in Gazebo
# validated=False → geometrically safe (min_wall=0.467m) but not yet Gazebo-tested
SPAWNS = [
    (-2.9, -2.0, 1.571, "P1", "#1a53ff", True),   # left channel heading N
    ( 1.0, -1.0, 1.571, "P2", "#ff6600", False),  # inner chamber heading N
]

OPEN_SOUTH_Y = -3.81
ARROW_LEN = 0.55
WALL_W_DRAW = 0.15


def wall_polygon(cx, cy, yaw, length):
    hl, hw = length / 2, WALL_W_DRAW / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cx + hl*c - hw*s, cy + hl*s + hw*c],
        [cx + hl*c + hw*s, cy + hl*s - hw*c],
        [cx - hl*c + hw*s, cy - hl*s - hw*c],
        [cx - hl*c - hw*s, cy - hl*s + hw*c],
    ])


fig, ax = plt.subplots(figsize=(11, 12))
ax.set_facecolor("#f0f4f8")
fig.patch.set_facecolor("white")

# ── Open south zone ──
open_rect = patches.Rectangle((-5.5, -6.5), 11.0, abs(-6.5 - OPEN_SOUTH_Y),
                                facecolor="#ffe0e0", edgecolor="none", zorder=0,
                                alpha=0.7, label="Open zone (no walls)")
ax.add_patch(open_rect)
ax.axhline(OPEN_SOUTH_Y, color="#cc4444", linewidth=1.4, linestyle="--", zorder=3)
ax.text(3.8, OPEN_SOUTH_Y + 0.12, "maze boundary\ny = -3.81m",
        fontsize=7.5, color="#cc4444", ha="right", va="bottom")

# ── Draw walls ──
for name, cx, cy, yaw, length in WALLS:
    poly = wall_polygon(cx, cy, yaw, length)
    face = "#cc0000" if name == "Wall_20" else "#555555"
    edge = "#aa0000" if name == "Wall_20" else "#333333"
    lw   = 1.2      if name == "Wall_20" else 0.6
    ax.add_patch(patches.Polygon(poly, closed=True, facecolor=face,
                                  edgecolor=edge, linewidth=lw, zorder=2))
    if name == "Wall_20":
        ax.text(cx + 0.15, cy + 0.25, "Wall_20", fontsize=6.5,
                color="#cc0000", ha="left", va="bottom", zorder=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cc0000", lw=0.8))

# ── Draw spawn points ──
for x, y, yaw, label, color, validated in SPAWNS:
    dx = np.cos(yaw) * ARROW_LEN
    dy = np.sin(yaw) * ARROW_LEN
    marker = "o" if validated else "x"
    if validated:
        ax.plot(x, y, marker="o", markersize=14, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=2.0, zorder=6)
    else:
        ax.plot(x, y, marker="x", markersize=13, markeredgewidth=2.5,
                color=color, zorder=6)
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.2, mutation_scale=14), zorder=7)
    val_str = "validated" if validated else "da validare"
    ax.text(x + 0.4, y, f"{label}\n{val_str}", fontsize=8, color=color,
            ha="left", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.0, alpha=0.95))

# ── Legend ──
legend_items = [
    patches.Patch(facecolor="#ffe0e0", edgecolor="#cc4444",
                  label="Open zone (no walls, y < -3.81m)"),
    Line2D([0],[0], marker="o", color="#1a53ff", markersize=10,
           markeredgecolor="white", linestyle="None",
           label="P1 (-2.9, -2.0) heading N — validated"),
    Line2D([0],[0], marker="x", color="#ff6600", markersize=10,
           markeredgewidth=2.5, linestyle="None",
           label="P2 ( 1.0, -1.0) heading N — da validare"),
    patches.Patch(facecolor="#cc0000", edgecolor="#880000",
                  label="Wall_20 (interior obstacle)"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=8.5,
          framealpha=0.95, title="Spawn M1 — 2 punti", title_fontsize=9)

ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-6.5, 5.0)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.5, color="#aaaaaa")
ax.axhline(0, color="#bbbbbb", linewidth=0.5)
ax.axvline(0, color="#bbbbbb", linewidth=0.5)
ax.set_xlabel("x (m)", fontsize=11)
ax.set_ylabel("y (m)", fontsize=11)
ax.set_title(
    "Maze 1 — 2 spawn points selezionati\n"
    "P1: canale sinistro heading N | P2: camera interna heading N",
    fontsize=12, fontweight="bold"
)

import os
os.makedirs("analysis/plots", exist_ok=True)
out = "analysis/plots/maze1_spawn_map.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()

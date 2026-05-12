"""
Plot Maze 2 (labirinto_9b) top-down with validated spawn points.
Wall data from <state> section of labirinto_9b.world (world-frame poses).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ── Wall data: (cx, cy, yaw_rad, length) — world-frame from <state> section ──
WALLS = [
    ("W0",  -7.62244,  3.9469,   1.5708,   4.10000),
    ("W1",  -5.26246,  5.9219,  -0.008474, 4.87000),
    ("W2",  -1.61837,  5.1794,  -0.5006,   3.07738),
    ("W3",   0.880125, 5.51131,  0.713734,  3.31012),
    ("W4",   4.66453,  6.53571, -0.003831,  5.37004),
    ("W5",   7.28453,  3.89571, -1.56702,   5.45004),
    ("W6",   5.83453,  1.25571,  3.13484,   3.11007),
    ("W7",   4.35452, -0.084292,-1.5708,    2.81000),
    ("W8",   2.6766,  -1.15077,  2.9742,    3.55342),
    ("W9",   1.00867, -2.47725, -1.56466,   3.41006),
    ("W10",  0.473678,-5.03657, -2.07596,   2.31965),
    ("W11", -1.11628, -6.32001, -2.83213,   2.34415),
    ("W12", -3.19622, -6.26454,  2.78784,   2.39920),
    ("W13", -4.77297, -5.08745,  2.13811,   2.01757),
    ("W14", -5.28977, -3.36293,  1.60815,   2.02534),
    ("W15", -4.94587, -1.79598,  1.01115,   1.66365),
    ("W17", -6.36674,  1.61544, -0.261799,  2.75000),
    ("W19", -3.08822, -1.83382,  1.0472,    1.50000),
    ("W20", -2.75073, -0.244257, 1.5509,    2.16090),
    ("W21", -3.33573,  1.57007,  2.20105,   2.20310),
    ("W22", -4.88495,  2.65116,  2.88619,   2.14310),
    ("W23", -5.85916,  3.58792,  1.58498,   1.56014),
    ("W24", -4.55917,  4.30292,  0.007519,  2.81007),
    ("W25", -1.69032,  3.45292, -0.513923,  3.73017),
    ("W26",  1.21132,  3.72571,  0.717922,  3.71569),
    ("W27",  4.03411,  4.9085,   0.006849,  3.07007),
    ("W28",  5.49411,  3.9685,  -1.5708,    2.09000),
    ("W29",  4.04911,  2.9685,  -3.13467,   3.04000),
    ("W30",  2.60412,  1.7735,  -1.5708,    2.56000),
    ("W31",  1.03693,  0.839671, 2.9579,    3.33801),
    ("W32", -0.530258,-1.09416, -1.5708,    4.60000),
]

WALL_W = 0.15  # wall thickness

# ── Validated spawn points: (x, y, yaw, label, zone, min_lidar) ──
SPAWNS = [
    (-6.0,  0.0,  0.0,   "A1", "A", 1.352),
    (-6.5, -0.5,  0.0,   "A2", "A", 1.803),
    (-4.5,  0.5,  0.0,   "B1", "B", 0.995),
    (-4.0, -1.0,  1.571, "B2", "B", 0.523),
    (-4.5,  1.5,  2.356, "B3", "B", 0.497),
    (-2.5,  1.0,  0.0,   "C1", "C", 0.434),
    (-7.0,  5.0,  0.0,   "C2", "C", 0.860),
    (-2.0, -1.0,  0.785, "C3", "C", 0.795),
    ( 1.5,  0.0,  3.142, "D1", "D", 0.693),
    ( 0.5, -2.0,  1.571, "D2", "D", 0.430),
    ( 3.5,  0.5,  4.712, "D3", "D", 0.780),
    (-3.0,  3.0,  0.0,   "E1", "E", 0.890),
    ( 0.0,  3.5,  3.142, "E2", "E", 0.650),
    (-4.5, -3.5,  0.0,   "F1", "F", 1.162),
    (-1.5, -4.0,  1.571, "F2", "F", 1.008),
    ( 6.0,  6.0,  3.142, "F3", "F", 0.456),
]

ZONE_COLORS = {"A": "#e74c3c", "B": "#e67e22", "C": "#2ecc71",
               "D": "#3498db", "E": "#9b59b6", "F": "#1abc9c"}

ARROW_LEN = 0.55  # heading arrow length


def wall_polygon(cx, cy, yaw, length):
    hl, hw = length / 2, WALL_W / 2
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [cx + hl*c - hw*s, cy + hl*s + hw*c],
        [cx + hl*c + hw*s, cy + hl*s - hw*c],
        [cx - hl*c + hw*s, cy - hl*s - hw*c],
        [cx - hl*c - hw*s, cy - hl*s + hw*c],
    ])
    return corners


fig, ax = plt.subplots(figsize=(14, 12))
ax.set_facecolor("#f8f8f8")
fig.patch.set_facecolor("white")

# ── Draw walls ──
for name, cx, cy, yaw, length in WALLS:
    poly = wall_polygon(cx, cy, yaw, length)
    patch = patches.Polygon(poly, closed=True, facecolor="#555555",
                            edgecolor="#333333", linewidth=0.6, zorder=2)
    ax.add_patch(patch)

# ── Draw spawn points ──
for x, y, yaw, label, zone, min_lidar in SPAWNS:
    color = ZONE_COLORS[zone]
    ax.plot(x, y, marker="o", markersize=11, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=1.8, zorder=5)
    dx = np.cos(yaw) * ARROW_LEN
    dy = np.sin(yaw) * ARROW_LEN
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=1.6, mutation_scale=12),
                zorder=6)
    off_x = -np.sin(yaw) * 0.5
    off_y =  np.cos(yaw) * 0.5
    ax.text(x + off_x, y + off_y,
            f"{label}\n{min_lidar:.3f}m",
            fontsize=7, fontweight="bold", color=color,
            ha="center", va="center", zorder=7,
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec=color, lw=0.8, alpha=0.92))

# ── Zone legend ──
from matplotlib.lines import Line2D
legend_items = [
    Line2D([0], [0], marker="o", color=ZONE_COLORS[z], markersize=9,
           markeredgecolor="white", linestyle="None",
           label=f"Zone {z} ({sum(1 for s in SPAWNS if s[4]==z)} pts)")
    for z in "ABCDEF"
]
ax.legend(handles=legend_items, loc="lower right", fontsize=9,
          framealpha=0.9, title="Zone (label = min LIDAR)")

ax.set_xlim(-9, 9)
ax.set_ylim(-8, 8)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5, color="#aaaaaa")
ax.axhline(0, color="#cccccc", linewidth=0.5)
ax.axvline(0, color="#cccccc", linewidth=0.5)
ax.set_xlabel("x (m)", fontsize=11)
ax.set_ylabel("y (m)", fontsize=11)
ax.set_title("Maze 2 — 16 validated spawn points (arrows = heading)",
             fontsize=13, fontweight="bold")

os.makedirs("analysis/plots", exist_ok=True)
out = "analysis/plots/maze2_spawn_map.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()

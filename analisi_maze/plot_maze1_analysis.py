"""
Maze 1 — full analysis plot.
Shows: open-south zone, corridor widths, U-turn feasibility, recommended 2 spawn points.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

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
OPEN_SOUTH_Y = -3.81   # bottom tip of Wall_0 and Wall_10
V_LIN = 0.5
W_MAX = 0.8
ROBOT_WIDTH = 0.305
r_min = V_LIN / W_MAX
req_width = 2 * r_min + ROBOT_WIDTH  # 1.555m


def wall_polygon(cx, cy, yaw, length):
    hl, hw = length / 2, WALL_W / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cx + hl*c - hw*s, cy + hl*s + hw*c],
        [cx + hl*c + hw*s, cy + hl*s - hw*c],
        [cx - hl*c + hw*s, cy - hl*s - hw*c],
        [cx - hl*c - hw*s, cy - hl*s + hw*c],
    ])


fig, ax = plt.subplots(figsize=(11, 13))
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
    face = "#cc0000" if name == "Wall_20" else "#555555"
    edge = "#880000" if name == "Wall_20" else "#333333"
    lw   = 1.2      if name == "Wall_20" else 0.6
    poly = wall_polygon(cx, cy, yaw, length)
    ax.add_patch(patches.Polygon(poly, closed=True, facecolor=face,
                                  edgecolor=edge, linewidth=lw, zorder=2))

# ── Corridor width annotations ──
corridors = [
    ("Left\nchannel\n1.50m\nFAIL",  -3.629, -2.125, -2.5, "#d35400"),
    ("Centre-L\n1.61m OK",           -2.125, -0.513, -0.8, "#27ae60"),
    ("Centre-R\n2.25m OK",           -0.513,  1.736,  0.6, "#27ae60"),
    ("Right\nchannel\n1.49m\nFAIL",   1.736,  3.221,  2.5, "#d35400"),
]
for label, x1, x2, y_ann, color in corridors:
    mid = (x1 + x2) / 2
    width = abs(x2 - x1)
    # Double arrow
    ax.annotate("", xy=(x2, y_ann), xytext=(x1, y_ann),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.5))
    ax.text(mid, y_ann + 0.18, label, fontsize=6.5, color=color,
            ha="center", va="bottom", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, lw=0.7, alpha=0.9))

# ── U-turn circle overlaid on centre-right (to show it fits) ──
circle = plt.Circle((0.6, -0.8), r_min, color="#27ae60",
                    fill=False, linestyle=":", linewidth=1.5, zorder=4, alpha=0.7)
ax.add_patch(circle)
ax.text(0.6, -0.8, f"r={r_min:.2f}m\nU-turn", fontsize=6, color="#27ae60",
        ha="center", va="center", zorder=5)

# ── OLD bad spawn points (greyed out) ──
old_bad = [
    ("A1", -3.0, -5.0, 1.57,   "heading N\n(open zone)"),
    ("A4", -1.5, -5.0, 0.0,    "heading E\nREWARD HACK"),
]
for label, x, y, yaw, note in old_bad:
    ax.plot(x, y, marker="x", markersize=10, markeredgewidth=2.0,
            color="#aaaaaa", zorder=4, alpha=0.7)
    dx, dy = np.cos(yaw)*0.5, np.sin(yaw)*0.5
    ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color="#aaaaaa", lw=1.2,
                                mutation_scale=10), zorder=4)
    ax.text(x + 0.15, y + 0.15, f"{label}\n{note}", fontsize=6.5,
            color="#999999", ha="left", va="bottom")

# ── RECOMMENDED 2 spawn points ──
recommended = [
    ("P1\nB1", -2.9, -2.0,  np.pi/2,  "#1a53ff", "S->N traversal\n(left channel)"),
    ("P2\nB2", -2.9,  0.5, -np.pi/2,  "#ff6600", "N->S traversal\n(left channel)"),
]
ARROW_LEN = 0.55
for label, x, y, yaw, color, desc in recommended:
    ax.plot(x, y, marker="o", markersize=13, markerfacecolor=color,
            markeredgecolor="white", markeredgewidth=2.0, zorder=6)
    dx, dy = np.cos(yaw)*ARROW_LEN, np.sin(yaw)*ARROW_LEN
    ax.annotate("", xy=(x+dx, y+dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2,
                                mutation_scale=14), zorder=7)
    off_x = 0.35 if x > -3 else -0.35
    ax.text(x + off_x, y, f"{label}\n{desc}", fontsize=7.5, color=color,
            ha="left" if off_x > 0 else "right", va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, lw=1.0, alpha=0.95))

# ── Legend ──
legend_items = [
    patches.Patch(facecolor="#ffe0e0", edgecolor="#cc4444", label="Open zone (no walls, y < -3.81m)"),
    Line2D([0],[0], marker="o", color="#1a53ff", markersize=10,
           markeredgecolor="white", linestyle="None", label="P1: B1 (-2.9,-2.0) heading N"),
    Line2D([0],[0], marker="o", color="#ff6600", markersize=10,
           markeredgecolor="white", linestyle="None", label="P2: B2 (-2.9, 0.5) heading S"),
    Line2D([0],[0], marker="x", color="#aaaaaa", markersize=9,
           markeredgewidth=2, linestyle="None", label="Removed spawns (open zone or bad heading)"),
    Line2D([0],[0], color="#d35400", linewidth=2, label=f"FAIL: width < {req_width:.2f}m (no U-turn)"),
    Line2D([0],[0], color="#27ae60", linewidth=2, label=f"OK: width >= {req_width:.2f}m (U-turn possible)"),
    patches.Patch(facecolor="#cc0000", edgecolor="#880000", label="Wall_20 (interior, C3 would collide)"),
]
ax.legend(handles=legend_items, loc="lower right", fontsize=7.8,
          framealpha=0.95, title="Analysis", title_fontsize=8.5)

ax.set_xlim(-5.5, 5.5)
ax.set_ylim(-6.5, 5.0)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", linewidth=0.35, alpha=0.45, color="#aaaaaa")
ax.axhline(0, color="#cccccc", linewidth=0.5)
ax.axvline(0, color="#cccccc", linewidth=0.5)
ax.set_xlabel("x (m)", fontsize=11)
ax.set_ylabel("y (m)", fontsize=11)
ax.set_title(
    "Maze 1 — analisi completa\n"
    f"U-turn: r_min={r_min:.3f}m, richiesto={req_width:.3f}m | "
    "Canali lat. FAIL | Open south: y < -3.81m",
    fontsize=11, fontweight="bold"
)

import os
os.makedirs("analisi_maze/plots", exist_ok=True)
out = "analisi_maze/plots/maze1_analysis.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()

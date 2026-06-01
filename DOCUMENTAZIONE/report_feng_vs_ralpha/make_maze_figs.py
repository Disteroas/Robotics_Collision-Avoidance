"""Clean top-down maze figures (walls + spawn points) for the report.

M1 (labirinto_9a) and M2 (labirinto_9b) wall geometry is world-frame, taken from
the validated <state> sections (analisi_maze/plot_maze{1,2}_spawns.py).
M3 (labirinto_10) is parsed live from the .world file.
Spawn points are the real evaluation spawns: TEST_SPAWN_LISTS in usv_env.py
(M1=2, M2=6, M3=1). Output: latex/figures/maze{1,2,3}.png

Only numpy + matplotlib + stdlib. Host Python.
"""
import os
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "latex", "figures")
os.makedirs(OUT, exist_ok=True)

WALL_W = 0.15
WALL_FACE, WALL_EDGE = "#444444", "#222222"
SPAWN_C = "#d62728"
ARROW_LEN = 0.6

# ── M1 walls (world-frame) ── from plot_maze1_spawns.py
WALLS_M1 = [
    (-3.62886, -1.12723, 1.57080, 5.36043), (-3.80387, 1.49349, 3.14159, 0.50000),
    (-3.97886, 2.34777, 1.57080, 1.88957), (-3.80387, 3.21756, 0.00000, 0.50000),
    (-3.62886, 3.54857, 1.57080, 0.81203), (-2.77366, 3.86408, 0.00000, 1.86043),
    (-1.91068, 3.54857, -1.54738, 0.81221), (0.82634, 3.21756, 0.00000, 5.63957),
    (3.57113, 2.35553, -1.57080, 1.87406), (3.39614, 1.50900, 3.14159, 0.50000),
    (3.22113, -1.11947, -1.57080, 5.37594), (2.33584, -3.74020, -3.13283, 1.92066),
    (1.46606, -3.40918, 1.57080, 0.82754), (-0.24090, -3.07041, 3.13251, 3.56405),
    (-2.12459, -1.05744, 1.57080, 5.50000), (-0.19959, 1.61756, 0.00000, 4.00000),
    (1.73643, 0.07197, -1.56367, 3.24126), (0.61464, -1.46694, 3.13958, 2.41562),
    (-0.51269, -0.73215, 1.56328, 1.60622), (-0.00162, 0.00361, 0.00646, 1.16122),
]
SPAWNS_M1 = [(-2.9, -2.0, 1.571), (1.0, -1.0, 1.571)]
LIM_M1 = (-5.5, 5.5, -6.0, 5.0)

# ── M2 walls (world-frame) ── from plot_maze2_spawns.py
WALLS_M2 = [
    (-7.62244, 3.9469, 1.5708, 4.10000), (-5.26246, 5.9219, -0.008474, 4.87000),
    (-1.61837, 5.1794, -0.5006, 3.07738), (0.880125, 5.51131, 0.713734, 3.31012),
    (4.66453, 6.53571, -0.003831, 5.37004), (7.28453, 3.89571, -1.56702, 5.45004),
    (5.83453, 1.25571, 3.13484, 3.11007), (4.35452, -0.084292, -1.5708, 2.81000),
    (2.6766, -1.15077, 2.9742, 3.55342), (1.00867, -2.47725, -1.56466, 3.41006),
    (0.473678, -5.03657, -2.07596, 2.31965), (-1.11628, -6.32001, -2.83213, 2.34415),
    (-3.19622, -6.26454, 2.78784, 2.39920), (-4.77297, -5.08745, 2.13811, 2.01757),
    (-5.28977, -3.36293, 1.60815, 2.02534), (-4.94587, -1.79598, 1.01115, 1.66365),
    (-6.36674, 1.61544, -0.261799, 2.75000), (-3.08822, -1.83382, 1.0472, 1.50000),
    (-2.75073, -0.244257, 1.5509, 2.16090), (-3.33573, 1.57007, 2.20105, 2.20310),
    (-4.88495, 2.65116, 2.88619, 2.14310), (-5.85916, 3.58792, 1.58498, 1.56014),
    (-4.55917, 4.30292, 0.007519, 2.81007), (-1.69032, 3.45292, -0.513923, 3.73017),
    (1.21132, 3.72571, 0.717922, 3.71569), (4.03411, 4.9085, 0.006849, 3.07007),
    (5.49411, 3.9685, -1.5708, 2.09000), (4.04911, 2.9685, -3.13467, 3.04000),
    (2.60412, 1.7735, -1.5708, 2.56000), (1.03693, 0.839671, 2.9579, 3.33801),
    (-0.530258, -1.09416, -1.5708, 4.60000),
]
SPAWNS_M2 = [(-6.0, 0.0, 0.0), (-7.0, 5.0, 0.0), (3.5, -0.5, 1.5708),
             (-4.5, -3.5, 0.0), (-1.5, -4.0, 1.571), (6.0, 6.0, 3.142)]
LIM_M2 = (-9.0, 9.0, -8.0, 8.0)

SPAWNS_M3 = [(-2.5, -0.25, 0.0)]
LIM_M3 = (-6.0, 6.0, -5.0, 5.0)


def parse_m3_walls(path):
    root = ET.parse(path).getroot()
    muri = next(m for m in root.iter("model") if m.get("name") == "Muri10")
    mp = muri.find("pose")
    ox, oy = (float(v) for v in mp.text.split()[:2]) if mp is not None else (0.0, 0.0)
    walls = []
    for ln in muri.findall("link"):
        if not (ln.get("name") or "").startswith("Wall"):
            continue
        p = ln.find("pose")
        box = ln.find(".//box/size")
        if p is None or box is None:
            continue
        px, py, _, _, _, yaw = (float(v) for v in p.text.split())
        L = float(box.text.split()[0])
        walls.append((px + ox, py + oy, yaw, L))
    return walls


def wall_polygon(cx, cy, yaw, length):
    hl, hw = length / 2, WALL_W / 2
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cx + hl * c - hw * s, cy + hl * s + hw * c],
        [cx + hl * c + hw * s, cy + hl * s - hw * c],
        [cx - hl * c + hw * s, cy - hl * s - hw * c],
        [cx - hl * c - hw * s, cy - hl * s + hw * c],
    ])


def render(walls, spawns, lim, fname, title):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfbfb")
    for cx, cy, yaw, L in walls:
        ax.add_patch(patches.Polygon(wall_polygon(cx, cy, yaw, L), closed=True,
                                     facecolor=WALL_FACE, edgecolor=WALL_EDGE,
                                     linewidth=0.5, zorder=2))
    for x, y, yaw in spawns:
        ax.annotate("", xy=(x + np.cos(yaw) * ARROW_LEN, y + np.sin(yaw) * ARROW_LEN),
                    xytext=(x, y), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color="black", lw=1.6,
                                    mutation_scale=13))
        ax.plot(x, y, "o", markersize=9, markerfacecolor=SPAWN_C,
                markeredgecolor="white", markeredgewidth=1.4, zorder=5)
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[2], lim[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_edgecolor("#cccccc")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=4)
    fig.tight_layout()
    out = os.path.join(OUT, fname)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out, "| walls:", len(walls), "| spawns:", len(spawns))


if __name__ == "__main__":
    walls_m3 = parse_m3_walls(os.path.join(ROOT, "src", "my_usv", "worlds",
                                           "labirinto_10.world"))
    render(WALLS_M1, SPAWNS_M1, LIM_M1, "maze1.png", "M1")
    render(WALLS_M2, SPAWNS_M2, LIM_M2, "maze2.png", "M2")
    render(walls_m3, SPAWNS_M3, LIM_M3, "maze3.png", "M3 (held-out)")

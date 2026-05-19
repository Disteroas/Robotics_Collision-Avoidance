"""
maze2_geom_check.py — analisi geometrica Maze 2 (Muri_9b)

Funzioni:
  1. Parsa Muri_9b/model.sdf → estrae walls (cx, cy, yaw, length)
  2. Calcola distanza punto-segmento per ogni spawn point
  3. Cerca candidati replacement per D1 in zone D con min_dist >= 0.50m
  4. Plot Maze 2 con spawn attivi + D1 candidate

Solo lettura. Non tocca train/test pipeline.
"""
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO   = Path(__file__).resolve().parents[1]
SDF    = REPO / "src" / "my_usv" / "worlds" / "Muri_9b" / "model.sdf"
OUT    = REPO / "analisi_maze" / "plots" / "maze2_spawn_map_v2.png"
WALL_W = 0.15

CURRENT_SPAWNS = [
    (-6.0,  0.0,  0.0,   "A1"),
    (-7.0,  5.0,  0.0,   "C2"),
    (-4.5, -3.5,  0.0,   "F1"),
    (-1.5, -4.0,  1.571, "F2"),
    ( 6.0,  6.0,  3.142, "F3"),
]
REMOVED_D1 = (1.5, 0.0, 3.142, "D1")


def parse_sdf_walls(path):
    """Estrae walls: (cx_world, cy_world, yaw, length)."""
    text = path.read_text()
    model_pose_match = re.search(
        r"<model name='Muri9b'>\s*<pose>([-\d\.\s]+)</pose>", text)
    if not model_pose_match:
        sys.exit("model pose not found")
    mp = list(map(float, model_pose_match.group(1).split()))
    model_dx, model_dy = mp[0], mp[1]

    link_blocks = re.findall(
        r"<link name='(Wall_\d+)'>(.*?)</link>", text, flags=re.DOTALL)
    walls = []
    for name, body in link_blocks:
        size_m = re.search(r"<size>([\d\.\-\s]+)</size>", body)
        pose_m = re.search(r"<pose>([-\d\.\s]+)</pose>\s*$", body.strip())
        if not (size_m and pose_m):
            continue
        length = float(size_m.group(1).split()[0])
        pose_vals = list(map(float, pose_m.group(1).split()))
        cx, cy, yaw = pose_vals[0], pose_vals[1], pose_vals[5]
        walls.append((cx + model_dx, cy + model_dy, yaw, length, name))
    return walls


def point_segment_distance(px, py, ax, ay, bx, by):
    """Distanza euclidea minima fra punto (px,py) e segmento (ax,ay)-(bx,by)."""
    dx, dy = bx - ax, by - ay
    L2 = dx*dx + dy*dy
    if L2 < 1e-9:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    cx, cy = ax + t * dx, ay + t * dy
    return math.hypot(px - cx, py - cy)


def min_distance_to_walls(px, py, walls):
    """min distanza da qualsiasi muro, considerando spessore WALL_W."""
    best = float("inf")
    for cx, cy, yaw, length, _ in walls:
        c, s = math.cos(yaw), math.sin(yaw)
        ax = cx - (length / 2) * c
        ay = cy - (length / 2) * s
        bx = cx + (length / 2) * c
        by = cy + (length / 2) * s
        d = point_segment_distance(px, py, ax, ay, bx, by) - WALL_W / 2
        best = min(best, max(0.0, d))
    return best


def search_d1_replacement(walls, target_zone_x=(0.0, 4.0),
                          target_zone_y=(-2.5, 1.5), min_clearance=0.55):
    """Grid search per replacement D1 in zona D centro-destra."""
    candidates = []
    xs = np.arange(target_zone_x[0], target_zone_x[1] + 0.01, 0.25)
    ys = np.arange(target_zone_y[0], target_zone_y[1] + 0.01, 0.25)
    for x in xs:
        for y in ys:
            d = min_distance_to_walls(x, y, walls)
            if d >= min_clearance:
                candidates.append((float(x), float(y), float(d)))
    candidates.sort(key=lambda c: -c[2])
    return candidates


def wall_polygon(cx, cy, yaw, length):
    hl, hw = length / 2, WALL_W / 2
    c, s = math.cos(yaw), math.sin(yaw)
    corners = np.array([
        [cx + hl*c - hw*s, cy + hl*s + hw*c],
        [cx + hl*c + hw*s, cy + hl*s - hw*c],
        [cx - hl*c + hw*s, cy - hl*s - hw*c],
        [cx - hl*c - hw*s, cy - hl*s + hw*c],
    ])
    return corners


def main():
    walls = parse_sdf_walls(SDF)
    print(f"Loaded {len(walls)} walls from Muri_9b/model.sdf")
    print()

    print("=== Min-distance attuali ===")
    rows = [REMOVED_D1] + [(x, y, yaw, lab) for x, y, yaw, lab in CURRENT_SPAWNS]
    for x, y, yaw, lab in rows:
        d = min_distance_to_walls(x, y, walls)
        status = "OK" if d >= 0.40 else ("WARNING" if d >= 0.25 else "COLLISION")
        print(f"  {lab:5s} ({x:+6.2f},{y:+6.2f}) yaw={math.degrees(yaw):+7.1f}°  min={d:.3f}m  [{status}]")
    print()

    print("=== D1 replacement search (zona D centro-destra) ===")
    candidates = search_d1_replacement(walls)
    print(f"  Trovati {len(candidates)} candidati con clearance >= 0.55m")
    print(f"  Top 8 per clearance:")
    for x, y, d in candidates[:8]:
        print(f"    ({x:+6.2f},{y:+6.2f})  min={d:.3f}m")

    chosen = candidates[0] if candidates else None
    print()
    if chosen:
        cx, cy, cd = chosen
        print(f"  Scelta: ({cx:+.2f},{cy:+.2f}), clearance={cd:.3f}m")
        print(f"  Yaw consigliato: TBD (cerca direzione corridoio piu' lungo)")

    print()
    print("=== Plot ===")
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_facecolor("#f8f8f8")
    fig.patch.set_facecolor("white")

    for cx, cy, yaw, length, name in walls:
        poly = wall_polygon(cx, cy, yaw, length)
        ax.add_patch(patches.Polygon(poly, closed=True,
                                     facecolor="#555555", edgecolor="#222",
                                     linewidth=0.5, zorder=2))

    rx, ry, ryaw, rlab = REMOVED_D1
    ax.plot(rx, ry, marker="x", markersize=14, color="red",
            markeredgewidth=3, zorder=4)
    ax.text(rx, ry - 0.55, f"{rlab} RIMOSSO", fontsize=8, color="red",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", fc="white",
                      ec="red", lw=0.8, alpha=0.9))

    for x, y, yaw, lab in CURRENT_SPAWNS:
        d = min_distance_to_walls(x, y, walls)
        color = "#27ae60"
        ax.plot(x, y, marker="o", markersize=12, markerfacecolor=color,
                markeredgecolor="white", markeredgewidth=1.8, zorder=5)
        dx_a = math.cos(yaw) * 0.55
        dy_a = math.sin(yaw) * 0.55
        ax.annotate("", xy=(x + dx_a, y + dy_a), xytext=(x, y),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                    lw=1.6, mutation_scale=12), zorder=6)
        ax.text(x - math.sin(yaw)*0.5, y + math.cos(yaw)*0.5,
                f"{lab}\n{d:.2f}m", fontsize=7, fontweight="bold",
                color=color, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.12", fc="white",
                          ec=color, lw=0.7, alpha=0.92), zorder=7)

    if candidates:
        for x, y, d in candidates[:8]:
            ax.plot(x, y, marker="s", markersize=8, markerfacecolor="#f39c12",
                    markeredgecolor="white", markeredgewidth=1.0, zorder=4, alpha=0.7)
        cx, cy, cd = chosen
        ax.plot(cx, cy, marker="*", markersize=22, markerfacecolor="#e67e22",
                markeredgecolor="black", markeredgewidth=1.5, zorder=8)
        ax.text(cx, cy - 0.7, f"D1_NEW?\n({cx:+.2f},{cy:+.2f})\n{cd:.2f}m",
                fontsize=8, fontweight="bold", color="#e67e22",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.18", fc="white",
                          ec="#e67e22", lw=1.0, alpha=0.95), zorder=9)

    legend = [
        Line2D([0], [0], marker="o", markerfacecolor="#27ae60",
               markeredgecolor="white", color="w", markersize=10,
               label=f"Active spawn ({len(CURRENT_SPAWNS)} pts)"),
        Line2D([0], [0], marker="x", color="red", markersize=12,
               markeredgewidth=3, linestyle="None", label="D1 RIMOSSO"),
        Line2D([0], [0], marker="*", markerfacecolor="#e67e22",
               markeredgecolor="black", color="w", markersize=14,
               label="D1 replacement candidate"),
        Line2D([0], [0], marker="s", markerfacecolor="#f39c12",
               markeredgecolor="white", color="w", markersize=8,
               label="Top 8 candidati"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=9, framealpha=0.92)

    ax.set_xlim(-9, 9)
    ax.set_ylim(-8, 8)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.axhline(0, color="#ccc", linewidth=0.5)
    ax.axvline(0, color="#ccc", linewidth=0.5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Maze 2 (Muri_9b) — 5 spawn attivi + ricerca D1 replacement",
                 fontsize=13, fontweight="bold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

"""Beats A+B — the metric (500 steps = full training-length coverage) read over
the results table being built, then the M2 'fair fight' row 32.5 -> 42.6."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import slide, vfx, style, scene12_data as sd

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FPS, SECONDS = 24, 21          # A (~13s) + B (~8s)
N = FPS * SECONDS


def main():
    m2_f, m2_r = sd.maze_mean("Feng", 2), sd.maze_mean("r_alpha", 2)
    m3_f, m3_r = sd.maze_mean("Feng", 3), sd.maze_mean("r_alpha", 3)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    slide.bg(ax)
    eb = slide.eyebrow(ax, 1.0, 8.4, "Objective: survive 500 steps = full training-length coverage")
    h1 = slide.headline(ax, 1.0, 7.6, "On the training maze, a fair fight.", size=28)

    # header
    cF = ax.text(8.6, 6.6, "Feng", ha="center", color=slide.RED, fontsize=20,
                 fontweight="bold", family=slide.FONT, alpha=0)
    cR = ax.text(12.4, 6.6, "r_alpha", ha="center", color=slide.BLUE, fontsize=20,
                 fontweight="bold", family=slide.FONT, alpha=0)
    # M2 row
    rlab = ax.text(2.0, 5.2, "M2 (train)", ha="left", color=slide.INK, fontsize=20,
                   fontweight="bold", family=slide.FONT, alpha=0)
    vF = ax.text(8.6, 5.2, "", ha="center", color=slide.RED, fontsize=24, fontweight="bold",
                 family=slide.FONT)
    vR = ax.text(12.4, 5.2, "", ha="center", color=slide.BLUE, fontsize=24, fontweight="bold",
                 family=slide.FONT)
    cap = ax.text(8, 1.2, "A small, honest win.", ha="center", color=slide.INK,
                  fontsize=20, family=slide.FONT, alpha=0)

    def main_count(target, t):
        return f"{target * vfx.ease(t):.1f}%"

    def upd(f):
        eb.set_alpha(vfx.eased_ramp(f, 0, 20))
        h1.set_alpha(vfx.eased_ramp(f, 10, 30))
        cF.set_alpha(vfx.eased_ramp(f, 40, 60)); cR.set_alpha(vfx.eased_ramp(f, 40, 60))
        rlab.set_alpha(vfx.eased_ramp(f, 50, 70))
        # values count up during beat B
        tb = vfx.ramp(f, 70, 130)
        vF.set_text(main_count(m2_f, tb)); vF.set_alpha(vfx.ramp(f, 60, 80))
        vR.set_text(main_count(m2_r, tb)); vR.set_alpha(vfx.ramp(f, 60, 80))
        cap.set_alpha(vfx.eased_ramp(f, 150, 180))
        return ()

    os.makedirs(OUT, exist_ok=True)
    FuncAnimation(fig, upd, frames=N, blit=False).save(
        os.path.join(OUT, "anim_results_table.mp4"), writer=style.writer(FPS))
    fig.savefig(os.path.join(OUT, "anim_results_table_end.png"), dpi=120)
    plt.close(fig)
    print(f"wrote anim_results_table.mp4  (M2 {m2_f:.1f}->{m2_r:.1f}, M3 {m3_f:.1f}->{m3_r:.1f})")


if __name__ == "__main__":
    main()

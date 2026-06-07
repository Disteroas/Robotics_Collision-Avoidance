"""Beat C — M3 is bimodal: per-seed dot plot, Feng vs r_alpha. We count seeds,
we don't average (report: IQM 'describes no actual run' on bimodal M3)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import slide, vfx, style, scene12_data as sd

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FPS, SECONDS = 24, 14
N = FPS * SECONDS


def panel(ax, x0, vals, color, label, count_text):
    """Scatter 10 seeds as a vertical jittered column at x0 (0..100%)."""
    ys = np.array([v * 100 for _, v in vals])
    xs = x0 + np.linspace(-0.5, 0.5, len(ys))
    sc = ax.scatter(xs, ys, s=140, color=color, edgecolor="white", lw=1.2,
                    zorder=5, alpha=0)
    lab = slide.headline(ax, x0, -8, label, size=18, ha="center", color=color)
    cnt = ax.text(x0, 112, count_text, ha="center", va="center", fontsize=22,
                  fontweight="bold", color=color, family=slide.FONT, alpha=0, zorder=5)
    return sc, lab, cnt, ys, xs


def main():
    feng = sd.m3_success_by_seed("Feng")
    ral = sd.m3_success_by_seed("r_alpha")

    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    slide.bg(ax)
    eb = slide.eyebrow(ax, 1.0, 8.4, "Unseen maze M3 — held out")
    h1 = slide.headline(ax, 1.0, 7.7, "Both agents are bimodal.", size=30)

    # data axes inset (success 0..100 mapped into the lower canvas)
    dax = fig.add_axes([0.12, 0.12, 0.76, 0.62]); dax.set_xlim(0, 3); dax.set_ylim(-14, 118)
    dax.axis("off")
    dax.axhline(0, color="#cfd8e3", lw=1.0); dax.axhline(100, color="#cfd8e3", lw=1.0)
    dax.text(-0.05, 0, "0%", ha="right", va="center", color=slide.SUB, fontsize=12)
    dax.text(-0.05, 100, "100%", ha="right", va="center", color=slide.SUB, fontsize=12)
    ylab = dax.text(-0.30, 50, "Success rate [%]", rotation=90, ha="center",
                    va="center", color=slide.SUB, fontsize=13, family=slide.FONT,
                    alpha=0)
    f_sc, f_lab, f_cnt, f_ys, f_xs = panel(dax, 1.0, feng, slide.RED, "Feng", "3 / 10")
    r_sc, r_lab, r_cnt, r_ys, r_xs = panel(dax, 2.0, ral, slide.BLUE, "r_alpha", "7 / 10")
    overlay = ax.text(8, 1.0, "We don't average. We count.", ha="center", va="center",
                      fontsize=18, color=slide.INK, family=slide.FONT, alpha=0, zorder=6)

    def upd(f):
        a_title = vfx.eased_ramp(f, 0, 18)
        eb.set_alpha(a_title); h1.set_alpha(a_title)
        # dots rise from 0 baseline into place
        a_dots = vfx.eased_ramp(f, 18, 70)
        for sc, ys, xs in ((f_sc, f_ys, f_xs), (r_sc, r_ys, r_xs)):
            sc.set_offsets(np.c_[xs, ys * a_dots])
            sc.set_alpha(vfx.ramp(f, 18, 40))
        for art in (f_lab, r_lab):
            art.set_alpha(vfx.eased_ramp(f, 30, 60))
        ylab.set_alpha(vfx.eased_ramp(f, 0, 30))
        for art in (f_cnt, r_cnt):
            art.set_alpha(vfx.eased_ramp(f, 80, 110))
        overlay.set_alpha(vfx.eased_ramp(f, 120, 150))
        return ()

    os.makedirs(OUT, exist_ok=True)
    anim = FuncAnimation(fig, upd, frames=N, blit=False)
    anim.save(os.path.join(OUT, "anim_bimodal_m3.mp4"), writer=style.writer(FPS))
    fig.savefig(os.path.join(OUT, "anim_bimodal_m3_end.png"), dpi=120)
    plt.close(fig)
    print("wrote anim_bimodal_m3.mp4")


if __name__ == "__main__":
    main()

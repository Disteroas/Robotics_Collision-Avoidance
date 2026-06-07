"""Beats A+B — the metric (500 steps = full training-length coverage) read over
the results table being built, the M2 'fair fight' as grouped bars (32.5 vs
42.6), then the M3 'averages lie' beat: M3 bars rise to 29 / 59.3, hold, then
dissolve into a question mark (the average is misleading on the bimodal unseen
maze — beat C then gives the honest count)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

import slide, vfx, style, scene12_data as sd

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FPS, SECONDS = 24, 24
N = FPS * SECONDS
BW = 1.2  # bar width (dax data units)


def bar(dax, x, color):
    r = Rectangle((x, 0), BW, 0, facecolor=color, edgecolor="white", lw=1.2,
                  alpha=0, zorder=4)
    dax.add_patch(r)
    return r


def main():
    m2_f, m2_r = sd.maze_mean("Feng", 2), sd.maze_mean("r_alpha", 2)
    m3_f, m3_r = sd.maze_mean("Feng", 3), sd.maze_mean("r_alpha", 3)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    slide.bg(ax)
    eb = slide.eyebrow(ax, 1.0, 8.4, "Objective: survive 500 steps = full training-length coverage")
    h1 = slide.headline(ax, 1.0, 7.6, "On the training maze, a fair fight.", size=28)

    # legend swatches (top-right)
    lf = ax.text(11.2, 8.4, "Feng", ha="left", color=slide.RED, fontsize=18,
                 fontweight="bold", family=slide.FONT, alpha=0)
    lr = ax.text(13.4, 8.4, "r_alpha", ha="left", color=slide.BLUE, fontsize=18,
                 fontweight="bold", family=slide.FONT, alpha=0)

    # bar chart axes
    dax = fig.add_axes([0.10, 0.13, 0.80, 0.52]); dax.axis("off")
    dax.set_xlim(0, 10); dax.set_ylim(-14, 112)
    dax.axhline(0, color="#cfd8e3", lw=1.0); dax.axhline(100, color="#cfd8e3", lw=1.0)
    dax.text(-0.15, 0, "0%", ha="right", va="center", color=slide.SUB, fontsize=12)
    dax.text(-0.15, 100, "100%", ha="right", va="center", color=slide.SUB, fontsize=12)
    dax.text(-0.7, 50, "Success rate [%]", rotation=90, ha="center", va="center",
             color=slide.SUB, fontsize=13, family=slide.FONT)

    xb = dict(m2f=1.6, m2r=3.0, m3f=6.0, m3r=7.4)
    b_m2f, b_m2r = bar(dax, xb["m2f"], slide.RED), bar(dax, xb["m2r"], slide.BLUE)
    b_m3f, b_m3r = bar(dax, xb["m3f"], slide.RED), bar(dax, xb["m3r"], slide.BLUE)

    def vlab(x):
        return dax.text(x + BW / 2, 0, "", ha="center", va="bottom", fontsize=18,
                        fontweight="bold", color=slide.INK, family=slide.FONT, alpha=0)
    t_m2f, t_m2r = vlab(xb["m2f"]), vlab(xb["m2r"])
    t_m3f, t_m3r = vlab(xb["m3f"]), vlab(xb["m3r"])

    g_m2 = dax.text((xb["m2f"] + xb["m2r"]) / 2 + BW / 2, -8, "M2 (train)", ha="center",
                    va="center", color=slide.INK, fontsize=18, fontweight="bold",
                    family=slide.FONT, alpha=0)
    g_m3 = dax.text((xb["m3f"] + xb["m3r"]) / 2 + BW / 2, -8, "M3 (unseen)", ha="center",
                    va="center", color=slide.INK, fontsize=18, fontweight="bold",
                    family=slide.FONT, alpha=0)

    qm = dax.text((xb["m3f"] + xb["m3r"]) / 2 + BW / 2, 55, "?", ha="center",
                  va="center", color=slide.INK, fontsize=90, fontweight="bold",
                  family=slide.FONT, alpha=0, zorder=6)

    cap1 = ax.text(8, 0.9, "A small, honest win.", ha="center", color=slide.INK,
                   fontsize=20, family=slide.FONT, alpha=0)
    cap2 = slide.headline(ax, 8, 0.9, "But the unseen test maze? That's where averages lie.",
                          size=22, ha="center")

    def grow(b, lbl, target, t):
        h = target * vfx.ease(t)
        b.set_height(h)
        lbl.set_position((b.get_x() + BW / 2, h + 2))
        lbl.set_text(f"{h:.1f}%")

    def upd(f):
        eb.set_alpha(vfx.eased_ramp(f, 0, 20))
        h1.set_alpha(vfx.eased_ramp(f, 10, 30))
        a_leg = vfx.eased_ramp(f, 40, 60)
        lf.set_alpha(a_leg); lr.set_alpha(a_leg)

        # M2 group
        g_m2.set_alpha(vfx.eased_ramp(f, 50, 70))
        a_m2 = vfx.ramp(f, 60, 80)
        for art in (b_m2f, b_m2r, t_m2f, t_m2r):
            art.set_alpha(a_m2)
        tb = vfx.ramp(f, 70, 150)
        grow(b_m2f, t_m2f, m2_f, tb); grow(b_m2r, t_m2r, m2_r, tb)
        cap1.set_alpha(vfx.eased_ramp(f, 160, 190) * (1 - vfx.ramp(f, 300, 320)))

        # M3 group: grow to value, hold, then dissolve
        g_m3.set_alpha(vfx.eased_ramp(f, 300, 320))
        a_m3 = vfx.ramp(f, 300, 320) * (1 - vfx.eased_ramp(f, 430, 470))
        for art in (b_m3f, b_m3r, t_m3f, t_m3r):
            art.set_alpha(a_m3)
        tb3 = vfx.ramp(f, 310, 390)
        grow(b_m3f, t_m3f, m3_f, tb3); grow(b_m3r, t_m3r, m3_r, tb3)

        # question mark + caption swap
        qm.set_alpha(vfx.eased_ramp(f, 440, 480))
        cap2.set_alpha(vfx.eased_ramp(f, 440, 480))
        return ()

    os.makedirs(OUT, exist_ok=True)
    FuncAnimation(fig, upd, frames=N, blit=False).save(
        os.path.join(OUT, "anim_results_table.mp4"), writer=style.writer(FPS))
    fig.savefig(os.path.join(OUT, "anim_results_table_end.png"), dpi=120)
    plt.close(fig)
    print(f"wrote anim_results_table.mp4  (M2 {m2_f:.1f}->{m2_r:.1f}, M3 {m3_f:.1f}->{m3_r:.1f})")


if __name__ == "__main__":
    main()

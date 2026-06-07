"""Beat D — same code, different machine: one bar travels hw_A -> hw_B while it
collapses 59% -> 0% on the held-out maze M3. Numbers hardcoded from the report:
hw_B has no per-seed CSV, only the reported aggregate."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation

import slide, vfx, style

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FPS, SECONDS = 24, 7
N = FPS * SECONDS
RED_ALERT = "#d62728"
HW_A, HW_B = 59.0, 0.0       # report M3 (hardcoded — no per-seed hw_B CSV)
BW = 1.4
X_A, X_B = 1.4, 7.2          # bar left-x at machine A / machine B
FLOOR = 2.5                  # collapsed bar rests as a flat red line on the baseline


def main():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    slide.bg(ax)
    eb = slide.eyebrow(ax, 1.0, 8.4, "Reproducibility — cross-machine")

    dax = fig.add_axes([0.12, 0.16, 0.76, 0.50]); dax.axis("off")
    dax.set_xlim(0, 10); dax.set_ylim(-16, 112)
    dax.axhline(0, color="#cfd8e3", lw=1.2)
    dax.text(-0.2, 0, "0%", ha="right", va="center", color=slide.SUB, fontsize=12)
    dax.text(-0.2, 100, "100%", ha="right", va="center", color=slide.SUB, fontsize=12)
    dax.text(-0.9, 50, "M3 success rate [%]", rotation=90, ha="center", va="center",
             color=slide.SUB, fontsize=13, family=slide.FONT)

    bar = Rectangle((X_A, 0), BW, HW_A, facecolor=slide.BLUE, edgecolor="none",
                    lw=0, alpha=0, zorder=4)
    dax.add_patch(bar)
    val = dax.text(X_A + BW / 2, HW_A + 3, "", ha="center", va="bottom", fontsize=22,
                   fontweight="bold", color=slide.INK, family=slide.FONT, alpha=0, zorder=5)

    # static machine labels under the baseline
    dax.text(X_A + BW / 2, -9, "hw_A", ha="center", va="center", color=slide.BLUE,
             fontsize=18, fontweight="bold", family=slide.FONT)
    dax.text(X_B + BW / 2, -9, "hw_B", ha="center", va="center", color=RED_ALERT,
             fontsize=18, fontweight="bold", family=slide.FONT)

    redtext = slide.headline(ax, 8, 0.9, "Same code. Different PC.", size=26,
                             ha="center", color=RED_ALERT)

    def upd(f):
        eb.set_alpha(vfx.eased_ramp(f, 0, 15))
        a_in = vfx.eased_ramp(f, 10, 35)
        bar.set_alpha(a_in); val.set_alpha(a_in)
        t = vfx.eased_ramp(f, 45, 130)            # travel + collapse together
        x = X_A + (X_B - X_A) * t
        h = HW_A + (HW_B - HW_A) * t
        h_draw = max(h, FLOOR)                    # collapsed bar stays a flat red line
        bar.set_x(x); bar.set_height(h_draw)
        bar.set_facecolor(vfx.lerp_color(slide.BLUE, RED_ALERT, t))
        val.set_position((x + BW / 2, h_draw + 2)); val.set_text(f"{h:.0f}%")
        redtext.set_alpha(vfx.eased_ramp(f, 120, 150))
        return ()

    os.makedirs(OUT, exist_ok=True)
    FuncAnimation(fig, upd, frames=N, blit=False).save(
        os.path.join(OUT, "anim_hw_collapse.mp4"), writer=style.writer(FPS))
    fig.savefig(os.path.join(OUT, "anim_hw_collapse_end.png"), dpi=120)
    plt.close(fig)
    print("wrote anim_hw_collapse.mp4")


if __name__ == "__main__":
    main()

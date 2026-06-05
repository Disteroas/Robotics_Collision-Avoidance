"""Beat D — same code, different machine: held-out M3 collapses 59% -> 0%."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FuncAnimation

import slide, vfx, style

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FPS, SECONDS = 24, 7
N = FPS * SECONDS
RED_ALERT = "#d62728"


def window(ax, x, y, w, h, title, color):
    frame = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                           facecolor="#0e1726", edgecolor=color, lw=2.5, alpha=0, zorder=3)
    ax.add_patch(frame)
    cap = ax.text(x + w / 2, y - 0.4, title, ha="center", color=color, fontsize=18,
                  fontweight="bold", family=slide.FONT, alpha=0, zorder=4)
    val = ax.text(x + w / 2, y + h / 2, "", ha="center", va="center", color="white",
                  fontsize=40, fontweight="bold", family=slide.FONT, alpha=0, zorder=4)
    note = ax.text(x + w / 2, y + h * 0.22, "M3 (unseen)", ha="center", color="#9fb0c3",
                   fontsize=14, family=slide.FONT, alpha=0, zorder=4)
    return frame, cap, val, note


def main():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    slide.bg(ax)
    eb = slide.eyebrow(ax, 1.0, 8.4, "Reproducibility — cross-machine")
    fA = window(ax, 1.4, 2.6, 5.6, 3.6, "Machine A", slide.BLUE)
    fB = window(ax, 9.0, 2.6, 5.6, 3.6, "Machine B", RED_ALERT)
    redtext = slide.headline(ax, 8, 1.1, "Same code. Different PC.", size=26,
                             ha="center", color=RED_ALERT)

    def upd(f):
        eb.set_alpha(vfx.eased_ramp(f, 0, 15))
        for frame, cap, val, note in (fA, fB):
            a = vfx.eased_ramp(f, 10, 35)
            frame.set_alpha(a); cap.set_alpha(a); note.set_alpha(a)
        fA[2].set_text("59%"); fA[2].set_alpha(vfx.eased_ramp(f, 40, 70))
        fB[2].set_text("0%");  fB[2].set_alpha(vfx.eased_ramp(f, 70, 100))
        redtext.set_alpha(vfx.eased_ramp(f, 110, 140))
        return ()

    os.makedirs(OUT, exist_ok=True)
    FuncAnimation(fig, upd, frames=N, blit=False).save(
        os.path.join(OUT, "anim_hw_collapse.mp4"), writer=style.writer(FPS))
    fig.savefig(os.path.join(OUT, "anim_hw_collapse_end.png"), dpi=120)
    plt.close(fig)
    print("wrote anim_hw_collapse.mp4")


if __name__ == "__main__":
    main()

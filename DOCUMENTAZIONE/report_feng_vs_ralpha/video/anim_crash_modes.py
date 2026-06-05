"""Beat E — why it crashes. Two M2 episodes, placeholder Gazebo windows + a real
action strip (11 discrete steering cells, chosen action highlighted, advancing).
Left: perceptual (steers into wall, open space adjacent). Right: kinematic
(R_min pocket, can't turn)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.animation import FuncAnimation

import slide, vfx, style, scene12_data as sd

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FPS, SECONDS = 24, 11
N = FPS * SECONDS
NA = 11  # discrete actions


def strip(ax, x, y, w, h):
    """11-cell action strip; returns the list of cell patches (left=hard left)."""
    cells = []
    cw = w / NA
    for i in range(NA):
        c = Rectangle((x + i * cw, y), cw * 0.9, h, facecolor="#e6ebf2",
                      edgecolor="white", lw=1.0, alpha=0, zorder=4)
        ax.add_patch(c); cells.append(c)
    return cells


def col(ax, x, title, color, actions):
    win = FancyBboxPatch((x, 3.3), 5.6, 3.4, boxstyle="round,pad=0.02,rounding_size=0.05",
                         facecolor="#0e1726", edgecolor=color, lw=2.5, alpha=0, zorder=3)
    ax.add_patch(win)
    ph = ax.text(x + 2.8, 5.0, "Gazebo clip", ha="center", va="center", color="#5b6b7f",
                 fontsize=13, family=slide.FONT, style="italic", alpha=0, zorder=4)
    cap = slide.headline(ax, x + 2.8, 7.0, title, size=20, ha="center", color=color)
    cells = strip(ax, x + 0.2, 2.5, 5.2, 0.5)
    return dict(win=win, ph=ph, cap=cap, cells=cells, actions=actions, color=color)


def main():
    perc = sd.crash_episode("Feng", 0, "(-6.0,0.0)", maze=2)
    kin = sd.crash_episode("Feng", 0, "(-7.0,5.0)", maze=2)

    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 16); ax.set_ylim(0, 9)
    slide.bg(ax)
    eb = slide.eyebrow(ax, 1.0, 8.4, "Failure modes — what the policy actually does")
    L = col(ax, 1.4, "Perceptual  (~3/4)", slide.RED, perc["actions"])
    R = col(ax, 9.0, "Kinematic  (~1/4)", slide.BLUE, kin["actions"])
    cap = slide.headline(ax, 8, 0.9, "The problem was never the driving. It was the seeing.",
                         size=22, ha="center")

    def light_strip(c, f):
        a_in = vfx.eased_ramp(f, 30, 55)
        n = len(c["actions"])
        # playhead advances over the visible window of the clip
        prog = vfx.ramp(f, 55, 230)
        idx = min(n - 1, int(prog * (n - 1)))
        act = c["actions"][idx]
        for i, cell in enumerate(c["cells"]):
            cell.set_alpha(a_in)
            cell.set_facecolor(c["color"] if i == act else "#e6ebf2")

    def upd(f):
        eb.set_alpha(vfx.eased_ramp(f, 0, 15))
        for c in (L, R):
            a = vfx.eased_ramp(f, 10, 35)
            c["win"].set_alpha(a); c["ph"].set_alpha(a); c["cap"].set_alpha(a)
            light_strip(c, f)
        cap.set_alpha(vfx.eased_ramp(f, 200, 240))
        return ()

    os.makedirs(OUT, exist_ok=True)
    FuncAnimation(fig, upd, frames=N, blit=False).save(
        os.path.join(OUT, "anim_crash_modes.mp4"), writer=style.writer(FPS))
    fig.savefig(os.path.join(OUT, "anim_crash_modes_end.png"), dpi=120)
    plt.close(fig)
    print(f"wrote anim_crash_modes.mp4  (perc {len(perc['actions'])} steps, "
          f"kin {len(kin['actions'])} steps)")


if __name__ == "__main__":
    main()

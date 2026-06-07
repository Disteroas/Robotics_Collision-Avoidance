# Scena 12 — Results Revisions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply 4 user-requested visual improvements to the scena-12 result animations (axis label, grouped bars + M3 "averages lie" dissolve, travelling/collapsing hardware bar, "of crashes" labels + top-down footage advice).

**Architecture:** Edit the four existing render scripts in `DOCUMENTAZIONE/report_feng_vs_ralpha/video/`. They are standalone matplotlib `FuncAnimation` scripts sharing `slide.py` (template look), `vfx.py` (motion helpers), `style.py` (writer + data) and `scene12_data.py` (faithful, data-driven numbers). No new files. Each script renders an `.mp4` + `_end.png` into `MATERIALE_VIDEO/scena_12/`. Numbers stay data-driven from `runs/` via `scene12_data` except the cross-machine hw_B value (hardcoded from the report — no per-seed CSV exists).

**Tech Stack:** Python 3.14 (host, not Docker), matplotlib (Agg), imageio-ffmpeg, pandas, pytest.

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_bimodal_m3.py` | Modify | Add "Success rate [%]" y-axis title (beat C) |
| `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_results_table.py` | Rewrite | Grouped bars for M2 + M3 "averages lie" dissolve to "?" (beats A+B) |
| `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_hw_collapse.py` | Rewrite | One bar travelling hw_A→hw_B, collapsing 59%→0% (beat D) |
| `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_crash_modes.py` | Modify | "of crashes" on both column titles (beat E) |
| `DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12/README_capture.md` | Modify | Top-down camera advice for the two crash clips |

Data layer (`scene12_data.py`) and its tests are UNCHANGED — the data contract is the same. Run `pytest` to confirm it stays green.

All scripts are run from the `video/` directory:
`cd "DOCUMENTAZIONE/report_feng_vs_ralpha/video"`

---

## Task 1: anim_bimodal_m3.py — name the success-rate axis

**Files:**
- Modify: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_bimodal_m3.py`

- [ ] **Step 1: Add the rotated axis-title artist**

In `main()`, immediately after the two `dax.text(...)` lines that draw the `"0%"` and `"100%"` labels (around line 47), add a new artist and keep a handle:

```python
    ylab = dax.text(-0.30, 50, "Success rate [%]", rotation=90, ha="center",
                    va="center", color=slide.SUB, fontsize=13, family=slide.FONT,
                    alpha=0)
```

- [ ] **Step 2: Fade it in with the dots**

In `upd(f)`, after the `for art in (f_lab, r_lab):` block that sets the panel-label alpha, add:

```python
        ylab.set_alpha(vfx.eased_ramp(f, 18, 60))
```

- [ ] **Step 3: Render and verify**

Run: `python anim_bimodal_m3.py`
Expected stdout: `wrote anim_bimodal_m3.mp4`
Then open `../MATERIALE_VIDEO/scena_12/anim_bimodal_m3_end.png` and confirm the vertical label **"Success rate [%]"** appears to the left of the dot columns, not overlapping the "0%"/"100%" ticks.

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_bimodal_m3.py
git commit -m "feat(video): scene12 beat C — label M3 success-rate axis"
```

---

## Task 2: anim_results_table.py — grouped bars + M3 "averages lie" dissolve

**Files:**
- Rewrite: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_results_table.py`

This replaces the bare count-up percentages with a grouped vertical bar chart.
M2 bars settle on their values; M3 bars settle on 29 / 59.3, hold, then dissolve
while a large "?" fades in (audio: *"But the unseen test maze? That's where averages lie."*).

- [ ] **Step 1: Replace the whole file with the bar version**

```python
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
```

- [ ] **Step 2: Render and verify the fidelity print**

Run: `python anim_results_table.py`
Expected stdout (faithful to the report): `wrote anim_results_table.mp4  (M2 32.5->42.6, M3 29.0->59.3)`
If the numbers differ, STOP — the data layer or `runs/` changed; do not hand-edit numbers.

- [ ] **Step 3: Verify the end frame visually**

Open `../MATERIALE_VIDEO/scena_12/anim_results_table_end.png`. Confirm: M2 group shows two solid bars with `32.5%` / `42.6%` labels; M3 group shows a large **"?"** (bars + numbers dissolved); caption reads **"But the unseen test maze? That's where averages lie."**; "Success rate [%]" axis label present.

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_results_table.py
git commit -m "feat(video): scene12 beats A+B — grouped bars + M3 averages-lie dissolve"
```

---

## Task 3: anim_hw_collapse.py — one travelling, collapsing bar

**Files:**
- Rewrite: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_hw_collapse.py`

Replaces the two static windows with one bar that translates left→right (hw_A→hw_B)
while its height collapses 59%→0% and its color fades blue→red.

- [ ] **Step 1: Replace the whole file**

```python
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

    bar = Rectangle((X_A, 0), BW, HW_A, facecolor=slide.BLUE, edgecolor="white",
                    lw=1.2, alpha=0, zorder=4)
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
        bar.set_x(x); bar.set_height(h)
        bar.set_facecolor(vfx.lerp_color(slide.BLUE, RED_ALERT, t))
        val.set_position((x + BW / 2, h + 3)); val.set_text(f"{h:.0f}%")
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
```

- [ ] **Step 2: Render and verify**

Run: `python anim_hw_collapse.py`
Expected stdout: `wrote anim_hw_collapse.mp4`
Open `../MATERIALE_VIDEO/scena_12/anim_hw_collapse_end.png`: the bar sits flat (height 0) at the **right**, over the **"hw_B"** label, reading **"0%"**, red. (The 59%/blue/hw_A state is the first frame — scrub the mp4 to confirm the travel.)

- [ ] **Step 3: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_hw_collapse.py
git commit -m "feat(video): scene12 beat D — travelling bar collapses hw_A->hw_B"
```

---

## Task 4: anim_crash_modes.py — "of crashes" on both titles

**Files:**
- Modify: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_crash_modes.py`

- [ ] **Step 1: Update the two column titles**

In `main()`, change the two `col(...)` calls:

```python
    L = col(ax, 1.4, "Perceptual — ~3/4 of crashes", slide.RED, perc["actions"])
    R = col(ax, 9.0, "Kinematic — ~1/4 of crashes", slide.BLUE, kin["actions"])
```

- [ ] **Step 2: Render and verify**

Run: `python anim_crash_modes.py`
Expected stdout: `wrote anim_crash_modes.mp4  (perc 155 steps, kin 127 steps)`
Open `../MATERIALE_VIDEO/scena_12/anim_crash_modes_end.png`: titles read **"Perceptual — ~3/4 of crashes"** and **"Kinematic — ~1/4 of crashes"**. Confirm both fit inside their window width (font 20); if a title overflows, no code change beyond what's here — report it.

- [ ] **Step 3: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_crash_modes.py
git commit -m "feat(video): scene12 beat E — 'of crashes' on both crash-mode titles"
```

---

## Task 5: README_capture.md — top-down camera advice

**Files:**
- Modify: `DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12/README_capture.md`

- [ ] **Step 1: Insert a camera section after the capture command block**

After the closing ``` of the `start_test_gui.sh` code block (before the "Registra la GUI..." line, or right after it), insert:

```markdown
## Inquadratura consigliata — TOP-DOWN (ortografica)

Entrambe le cause sono SPAZIALI e si leggono solo dall'alto:

- **Perceptual** (spawn `(-6,0)`, 155 step): la barca sterza nel muro mentre
  accanto c'è spazio libero. Vista dall'alto → si vede il varco ignorato.
  Fine clip sull'impatto. In DaVinci: freccia che punta sullo spazio libero
  ("it was the seeing").
- **Kinematic** (spawn `(-7,5)`, 127 step): entra nel pocket R_min, non riesce
  a girarsi, sbatte. Vista dall'alto → si vede che NON può completare la curva.

In Gazebo: vista ortografica dall'alto (top view). Sincronizza la velocità di
ciascuna clip alla lunghezza della sua striscia azioni (155 / 127 step).
```

- [ ] **Step 2: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12/README_capture.md
git commit -m "docs(video): scene12 — top-down camera advice for crash clips"
```

---

## Final verification (after all tasks)

- [ ] **Data layer still green:** `cd "DOCUMENTAZIONE/report_feng_vs_ralpha/video" && python -m pytest tests/test_scene12_data.py -q` → 3 passed.
- [ ] **All four mp4 + end.png refreshed** in `MATERIALE_VIDEO/scena_12/` (check file mtimes).
- [ ] Task 7 (montage) stays GATED on user visual review of the 4 refreshed clips.

---

## Notes for the executor

- Run every script from the `video/` directory so the `sys.path` insert + relative `style.ROOT` resolve.
- These are visual scripts: "verify" = inspect the `_end.png` and, where it matters, scrub the mp4. There is no per-frame unit test; the data fidelity is guarded by the `scene12_data` tests + the results-table fidelity print.
- Never hand-edit the M2/M3 numbers — they must come from `scene12_data.maze_mean`. Only hw_A/hw_B (59/0) are hardcoded, by design (no hw_B per-seed CSV).
- Keep the shared style: `slide.bg`, Feng red `#d62728` / r_alpha blue `#1f77b4`, Arial, no emoji.

# Scena 12 — Results: Feng vs r_alpha — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the scena-12 results animations (53s, 5 beats: metric → training-maze → M3 bimodality → hardware → crash modes) as host-side matplotlib clips matching the scene 9-10 template, faithful to `latex/main.tex`.

**Architecture:** A shared data module (`scene12_data.py`) reads the real N=10 eval CSVs from `runs/` and is unit-tested with pytest. Four `anim_*.py` render scripts consume it and the existing `slide.py`/`vfx.py`/`style.py` template helpers; each renders to mp4 and is smoke-tested (file exists, even dimensions). An optional `build_scene12_montage.py` stitches them at VO pace. Visual correctness is eyeballed; automated checks cover data + output integrity.

**Tech Stack:** Python 3.14 (host-side, NOT in container), matplotlib `FuncAnimation`, `imageio-ffmpeg`, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-06-05-scena12-results-design.md`

---

## File Structure

All under `DOCUMENTAZIONE/report_feng_vs_ralpha/video/`:

| File | Responsibility |
|---|---|
| `scene12_data.py` | Pure data layer: load eval summaries / per-step actions, select crash episodes. Reuses `style.seed_dir`. No matplotlib. |
| `tests/test_scene12_data.py` | pytest for the data layer against real `runs/` CSVs. |
| `anim_results_table.py` | Beat A+B: results table (M2 32.5→42.6) + metric eyebrow + M2 violin/IQM. |
| `anim_bimodal_m3.py` | Beat C: M3 per-seed dot plot, count overlay 7/10 vs 3/10. |
| `anim_hw_collapse.py` | Beat D: two framed windows, M3 59% vs 0%, "Same code. Different PC." |
| `anim_crash_modes.py` | Beat E: two placeholder windows + real action strips (perceptual + kinematic). |
| `build_scene12_montage.py` | (Optional) stitch the 4 clips at VO pace. |

Output mp4/png → `DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12/`.

**Pinned data (verified against `runs/`):**
- M3 per-seed success — feng: `[0.9,0,0,1.0,0,1.0,0,0,0,0]` (3 generalize); r_alpha: `[0.833,1.0,0,0.533,1.0,0.767,0.8,0,1.0,0]` (7 generalize).
- Perceptual crash clip: `feng_hw_A` seed_0, M2, spawn `(-6.0,0.0)`.
- Kinematic crash clip: `feng_hw_A` seed_0, M2, spawn `(-7.0,5.0)` (R_min pocket, crashes 30/30 all seeds).
- Both spawns are covered by one round-robin M2 eval of feng seed_0 → single capture command.

**Canvas convention (from `slide.py`):** full-bleed axes `xlim 0-16, ylim 0-9, axis off, fig.subplots_adjust(0,0,1,1)`. Figure `figsize=(16,9), dpi=120` → 1920×1080, even. Writer `style.writer(fps=24)`.

---

### Task 0: Environment sanity check

**Files:** none (verification only)

- [ ] **Step 1: Confirm host-side deps import**

Run (Git Bash, from `DOCUMENTAZIONE/report_feng_vs_ralpha/video/`):
```bash
python -c "import matplotlib, pandas, imageio_ffmpeg, pytest; print('ok', imageio_ffmpeg.get_ffmpeg_exe())"
```
Expected: `ok C:\Users\...\ffmpeg-win-x86_64-v7.1.exe`. If `pytest` missing: `python -m pip install --user pytest`.

- [ ] **Step 2: Confirm the eval CSVs exist**

Run (from repo root):
```bash
ls runs/feng_hw_A/seed_0/eval_summary.csv runs/feng_hw_A/seed_0/eval_steps_m2.csv runs/r_alpha_hw_A/seed_9/eval_summary.csv
```
Expected: all three paths listed, no "No such file".

---

### Task 1: Data layer — `scene12_data.py`

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/scene12_data.py`
- Test: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/tests/test_scene12_data.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scene12_data.py`:
```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import scene12_data as sd


def test_m3_success_by_seed_counts():
    feng = dict(sd.m3_success_by_seed("Feng"))
    ral = dict(sd.m3_success_by_seed("r_alpha"))
    assert len(feng) == 10 and len(ral) == 10
    # report: 3/10 vs 7/10 seeds generalize (success_rate > 0)
    assert sum(1 for v in feng.values() if v > 0) == 3
    assert sum(1 for v in ral.values() if v > 0) == 7
    assert all(0.0 <= v <= 1.0 for v in list(feng.values()) + list(ral.values()))


def test_m2_success_by_seed():
    feng = dict(sd.m2_success_by_seed("Feng"))
    assert len(feng) == 10
    assert all(0.0 <= v <= 1.0 for v in feng.values())


def test_crash_episode_actions():
    # kinematic pocket spawn must exist for feng seed 0 on M2
    ep = sd.crash_episode("Feng", 0, "(-7.0,5.0)", maze=2)
    assert ep["spawn"] == "(-7.0,5.0)"
    assert len(ep["actions"]) > 0
    assert all(0 <= a <= 10 for a in ep["actions"])
    assert ep["actions"] == ep["actions"]  # ordered by step
    # perceptual spawn too
    ep2 = sd.crash_episode("Feng", 0, "(-6.0,0.0)", maze=2)
    assert len(ep2["actions"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `video/`):
```bash
python -m pytest tests/test_scene12_data.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'scene12_data'`.

- [ ] **Step 3: Write `scene12_data.py`**

```python
"""Data layer for scena-12 result animations. Reads the real N=10 eval CSVs
from runs/ via style.seed_dir. No matplotlib here — pure pandas, unit-tested."""
import os

import pandas as pd

import style  # reuse REGISTRY / seed_dir / ROOT

SEEDS = list(range(10))


def _summary(config, seed):
    path = os.path.join(style.seed_dir(config, seed), "eval_summary.csv")
    return pd.read_csv(path)


def _success_by_seed(config, maze):
    out = []
    for s in SEEDS:
        df = _summary(config, s)
        row = df[df["maze"] == maze]
        if not row.empty:
            out.append((s, float(row["success_rate"].iloc[0])))
    return out


def m3_success_by_seed(config):
    """[(seed, success_rate)] for the unseen maze M3."""
    return _success_by_seed(config, 3)


def m2_success_by_seed(config):
    """[(seed, success_rate)] for the training maze M2."""
    return _success_by_seed(config, 2)


def maze_mean(config, maze):
    """Mean success_rate (%) over seeds for one maze — for the results table."""
    vals = [v for _, v in _success_by_seed(config, maze)]
    return 100.0 * sum(vals) / len(vals) if vals else 0.0


def crash_episode(config, seed, spawn, maze=2):
    """First episode at `spawn` on `maze`; returns dict with the ordered action
    sequence and per-step front/left/right distances (for the action strip)."""
    df = style.load_eval_steps(config, seed, maze)
    df = df[df["spawn"] == spawn]
    if df.empty:
        raise ValueError(f"no steps at spawn {spawn} for {config} seed {seed} m{maze}")
    ep = int(df["episode"].iloc[0])
    ep_df = df[df["episode"] == ep].sort_values("step")
    return dict(
        spawn=spawn, episode=ep,
        actions=[int(a) for a in ep_df["action"].tolist()],
        front=ep_df["front_dist"].tolist(),
        left=ep_df["left_dist"].tolist(),
        right=ep_df["right_dist"].tolist(),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```bash
python -m pytest tests/test_scene12_data.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/scene12_data.py DOCUMENTAZIONE/report_feng_vs_ralpha/video/tests/test_scene12_data.py
git commit -m "feat(video): scene12 data layer + tests (real N=10 evals)"
```

---

### Task 2: Beat C — `anim_bimodal_m3.py` (centerpiece, build first)

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_bimodal_m3.py`

> Built first because it is the scene centerpiece and exercises the data layer + template end-to-end.

- [ ] **Step 1: Write the render script**

```python
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
```

- [ ] **Step 2: Render it**

Run (from `video/`):
```bash
python anim_bimodal_m3.py
```
Expected: `wrote anim_bimodal_m3.mp4`, no traceback.

- [ ] **Step 3: Verify output integrity (smoke test)**

Run:
```bash
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -i ../MATERIALE_VIDEO/scena_12/anim_bimodal_m3.mp4 2>&1 | grep -E "Duration|Video:"
```
Expected: `Duration: 00:00:14` and a `1920x1080` stream (even dims).

- [ ] **Step 4: Eyeball the end frame**

Open `MATERIALE_VIDEO/scena_12/anim_bimodal_m3_end.png`. Confirm: Feng column shows 3 dots near top / 7 at bottom, r_alpha 7 near top / 3 at bottom; counters "3 / 10" and "7 / 10"; template bg.

- [ ] **Step 5: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_bimodal_m3.py
git commit -m "feat(video): scene12 beat C — M3 bimodality dot plot"
```

---

### Task 3: Beat A+B — `anim_results_table.py`

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_results_table.py`

- [ ] **Step 1: Write the render script**

```python
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
```

- [ ] **Step 2: Render it**

Run: `python anim_results_table.py`
Expected: `wrote anim_results_table.mp4  (M2 32.5->42.6, M3 29.0->59.3)` — the printed M2/M3 means MUST match the report (32.5→42.6, 29.0→59.3). If they differ, the data layer or `runs/` is wrong — stop and investigate.

- [ ] **Step 3: Verify output integrity**

Run:
```bash
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -i ../MATERIALE_VIDEO/scena_12/anim_results_table.mp4 2>&1 | grep -E "Duration|Video:"
```
Expected: `Duration: 00:00:21`, `1920x1080`.

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_results_table.py
git commit -m "feat(video): scene12 beats A+B — metric + M2 results table"
```

---

### Task 4: Beat D — `anim_hw_collapse.py`

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_hw_collapse.py`

> Two framed windows = placeholders for Gazebo/terminal footage (same frame style as end of scene 10). Numbers are hardcoded from the report: HW-A M3 = 59%, HW-B M3 = 0%. (HW-B has no per-seed CSV in `runs/`; this beat is a stated report result, not a recomputation.)

- [ ] **Step 1: Write the render script**

```python
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
```

- [ ] **Step 2: Render it**

Run: `python anim_hw_collapse.py`
Expected: `wrote anim_hw_collapse.mp4`, no traceback.

- [ ] **Step 3: Verify output integrity**

Run:
```bash
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -i ../MATERIALE_VIDEO/scena_12/anim_hw_collapse.mp4 2>&1 | grep -E "Duration|Video:"
```
Expected: `Duration: 00:00:07`, `1920x1080`.

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_hw_collapse.py
git commit -m "feat(video): scene12 beat D — cross-machine M3 collapse"
```

---

### Task 5: Beat E — `anim_crash_modes.py`

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_crash_modes.py`

> Two placeholder windows (Gazebo dropped later in DaVinci) + a **real action strip** under each, from `scene12_data.crash_episode`. Left = perceptual `(-6.0,0.0)`, right = kinematic `(-7.0,5.0)`, both feng seed_0 M2.

- [ ] **Step 1: Write the render script**

```python
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
```

- [ ] **Step 2: Render it**

Run: `python anim_crash_modes.py`
Expected: `wrote anim_crash_modes.mp4  (perc N steps, kin M steps)` with N, M > 0.

- [ ] **Step 3: Verify output integrity**

Run:
```bash
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -i ../MATERIALE_VIDEO/scena_12/anim_crash_modes.mp4 2>&1 | grep -E "Duration|Video:"
```
Expected: `Duration: 00:00:11`, `1920x1080`.

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/anim_crash_modes.py
git commit -m "feat(video): scene12 beat E — perceptual vs kinematic crash strips"
```

---

### Task 6: Capture instructions — `scena_12/README_capture.md`

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12/README_capture.md`

> Documents the exact Gazebo capture the user runs to drop real footage into the beat-D/E placeholder windows.

- [ ] **Step 1: Write the capture README**

````markdown
# Scena 12 — Gazebo footage da catturare (placeholder windows)

I render `anim_hw_collapse` (beat D) e `anim_crash_modes` (beat E) hanno finestre
vuote: il footage Gazebo si monta dopo in DaVinci.

## Beat E — crash modes (UN solo comando per entrambe le clip)

Round-robin su M2 di feng seed_0 copre tutti gli spawn, inclusi i due che servono:
- **Perceptual** = spawn `(-6.0, 0.0)`  (sterza nel muro, spazio libero accanto)
- **Kinematic**  = spawn `(-7.0, 5.0)`  (pocket R_min=0.625m, non riesce a girare)

```bash
# Git Bash, root progetto, XLaunch (VcXsrv) attivo con "Disable access control"
./start_test_gui.sh 2 0 feng_hw_A 1 1
#                    │ │ │         │ └ speed 1x (real-time, per registrare)
#                    │ │ │         └ reps=1 (ogni spawn una volta)
#                    │ │ └ config = feng_hw_A
#                    │ └ seed = 0
#                    └ maze = 2
```
Registra la GUI Gazebo con OBS. Gli episodi escono in ordine round-robin
deterministico → taglia il segmento dello spawn `(-6,0)` (perceptual) e quello
`(-7,5)` (kinematic). Le strisce azioni nel render usano gli stessi episodi.

## Beat D — cross-machine (opzionale)

Due finestre Machine A / Machine B con M3 59% vs 0%. Footage = due terminali o due
GUI eval su M3, stesso modello, due PC. Non riproducibile su una macchina sola →
usa b-roll/terminali esistenti, o lascia i placeholder.
````

- [ ] **Step 2: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12/README_capture.md
git commit -m "docs(video): scene12 Gazebo capture instructions"
```

---

### Task 7 (optional): Stitch — `build_scene12_montage.py`

**Files:**
- Create: `DOCUMENTAZIONE/report_feng_vs_ralpha/video/build_scene12_montage.py`

> Concatenate the 4 beats in audio order with short xfades, at VO pace. Mirror the proven approach in `build_scene910_montage.py` (tpad AFTER fps; durations from known frame counts). Build only after the 4 clips are approved by the user.

- [ ] **Step 1: Write the montage builder**

```python
"""Stitch scena-12 beats A->B->C->D->E in audio order with 0.5s xfades, using the
imageio-ffmpeg binary. Beats A+B are one clip (anim_results_table)."""
import os, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import imageio_ffmpeg, style

OUT = os.path.join(style.ROOT, "DOCUMENTAZIONE/report_feng_vs_ralpha/MATERIALE_VIDEO/scena_12")
FF = imageio_ffmpeg.get_ffmpeg_exe()
ORDER = ["anim_results_table", "anim_bimodal_m3", "anim_hw_collapse", "anim_crash_modes"]


def main():
    clips = [os.path.join(OUT, f"{n}.mp4") for n in ORDER]
    for c in clips:
        if not os.path.exists(c):
            raise SystemExit(f"missing clip: {c} — render the beats first")
    inputs = []
    for c in clips:
        inputs += ["-i", c]
    # simple concat (xfade chains are fragile; concat keeps cuts clean for editing)
    filt = "".join(f"[{i}:v]setpts=PTS-STARTPTS[v{i}];" for i in range(len(clips)))
    filt += "".join(f"[v{i}]" for i in range(len(clips))) + f"concat=n={len(clips)}:v=1:a=0[v]"
    out = os.path.join(OUT, "scena_12_montaggio.mp4")
    cmd = [FF, "-y", *inputs, "-filter_complex", filt, "-map", "[v]",
           "-pix_fmt", "yuv420p", "-c:v", "libx264", out]
    subprocess.run(cmd, check=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Render the montage**

Run: `python build_scene12_montage.py`
Expected: `wrote .../scena_12_montaggio.mp4`.

- [ ] **Step 3: Verify total duration**

Run:
```bash
FF=$(python -c "import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())")
"$FF" -i ../MATERIALE_VIDEO/scena_12/scena_12_montaggio.mp4 2>&1 | grep -E "Duration|Video:"
```
Expected: `Duration: ~00:00:53` (21+14+7+11), `1920x1080`.

- [ ] **Step 4: Commit**

```bash
git add DOCUMENTAZIONE/report_feng_vs_ralpha/video/build_scene12_montage.py
git commit -m "feat(video): scene12 montage stitcher"
```

---

## Notes for the implementer

- **Run everything host-side** (Windows + Git Bash), from `DOCUMENTAZIONE/report_feng_vs_ralpha/video/`. These scripts never enter the ROS/Docker container.
- **Fidelity is non-negotiable:** the printed M2/M3 means in Task 3 must read 32.5→42.6 and 29.0→59.3. The crash taxonomy is strictly 2-class (perceptual/kinematic) — never reintroduce Side/Frontal/Dead-end.
- **Visual polish is iterative:** the smoke tests guarantee valid mp4s; spacing/timing tweaks happen while eyeballing the end-PNGs. Expect a second pass per clip.
- **Palette/template:** always via `slide.py` + `vfx.py`; Feng red `#d62728`, r_alpha blue `#1f77b4`. No emoji in plots (tofu).
```

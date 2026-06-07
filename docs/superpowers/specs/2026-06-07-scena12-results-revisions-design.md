# Scena 12 — Results revisions (design)

Date: 2026-06-07
Branch: paper_metric_feng
Supersedes nothing — refines the 4 render scripts shipped on 05/06
(`2026-06-05-scena12-results-design.md`).

## Goal

Improve the four scena-12 result animations per user direction, keeping every
number faithful to the report and to `MATERIALE_VIDEO/scena_12/video_scaletta_0506.md`
(segment 12 audio, 7:00 → 7:53, 53 s). No new files; edit the existing scripts.

Faithful numbers (do not invent):
- M2: Feng 32.5% → r_alpha 42.6% (data-driven via `scene12_data.maze_mean`).
- M3: Feng 29.0% → r_alpha 59.3% (data-driven).
- M3 per-seed generalization count: Feng 3/10, r_alpha 7/10 (`m3_success_by_seed`).
- Cross-machine M3: hw_A 59% → hw_B 0% (hardcoded — report; no per-seed CSV for hw_B).
- Crash taxonomy (report, 2-class): perceptual ~3/4, kinematic ~1/4. Audio voices
  only "three-quarters were perceptual"; 1/4 kinematic = the report complement.

Shared style unchanged: `slide.bg` white grid background + corner glow, Feng red
`#d62728` / r_alpha blue `#1f77b4`, INK headline, `slide.eyebrow`/`slide.headline`,
Arial, no emoji. Numbers come from `scene12_data`; tests extended where new numeric
artists are introduced.

## Changes

### 1. `anim_bimodal_m3.py` — name the axis

Add a rotated y-axis title **"Success rate [%]"** on the left of the `dax` inset
(vertical, color `slide.SUB`, fades in together with the dots via the existing
`a_dots` / dot alpha ramp). The 0% and 100% gridlines + tick labels already exist;
this only labels the axis so the scatter reads as success rate.

Everything else unchanged: per-seed scatter, "3 / 10" vs "7 / 10" count overlay,
"We don't average. We count." caption, 14 s.

### 2. `anim_results_table.py` — grouped bars + M3 "averages lie" dissolve

Replace the M2 count-up text values with a **grouped vertical bar chart** on an
inset data axes, y-axis 0–100% success. Keep beat A (metric) as-is.

- **Beat A (metric)** — unchanged: eyebrow "Objective: survive 500 steps = full
  training-length coverage", headline "On the training maze, a fair fight." held
  over the metric VO.
- **Beat B (M2 fair fight)** — group "M2 (train)": two bars grow from baseline,
  Feng red to `maze_mean("Feng",2)` (~32.5%), r_alpha blue to `maze_mean("r_alpha",2)`
  (~42.6%); numeric label `"NN.N%"` on top of each bar, counting up with the bar.
  Caption "A small, honest win."
- **Beat B-tail (M3, "averages lie")** — group "M3 (unseen)": two bars grow to
  `maze_mean("Feng",3)` (~29%) and `maze_mean("r_alpha",3)` (~59.3%) with numeric
  labels, **hold ~0.5 s**, then bars + their numeric labels **dissolve** (alpha → 0)
  while a large **"?"** (color `slide.INK`/`SUB`) fades in centered over the M3 group.
  Caption swaps to **"But the unseen test maze? That's where averages lie."**
  End frame: M2 group solid with numbers, M3 group = "?".

Layout: M2 group left, M3 group right, shared 0–100% axis with a Feng/r_alpha
legend (or colored header labels reused from current header artists). The clip
holds the long metric VO + M2 + M3 dissolve; default duration raised to ~24–26 s.
**Exact final duration is tuned during the montage (Task 7).**

Fidelity gate (print at end, unchanged contract): the script must print the
actual `maze_mean` values so a reviewer can confirm `M2 32.5->42.6, M3 29.0->59.3`.

### 3. `anim_hw_collapse.py` — single travelling, collapsing bar

Remove the two static `window(...)` panels. Draw one **vertical bar** on a
horizontal baseline spanning the canvas:

- t-start: bar at the **left** x-position, height ∝ **59%**, "59%" label on top,
  static label **"hw_A"** under the baseline at the left x-position.
- animation: bar **translates left → right** while its height interpolates
  **59% → 0%** and the top label counts **59 → 0** (eased). The collapse and the
  travel happen together over the main beat.
- t-end: bar flat (height 0) at the **right** x-position, "0%", static label
  **"hw_B"** under the baseline at the right x-position.
- Eyebrow "Reproducibility — cross-machine" stays; headline
  **"Same code. Different PC."** (red) fades in during the collapse.

Numbers hardcoded with the existing source comment (report; no per-seed hw_B CSV).
Duration ~7 s default (montage may retime).

### 4. `anim_crash_modes.py` — "of crashes" + footage advice

- Column titles → **"Perceptual — ~3/4 of crashes"** and
  **"Kinematic — ~1/4 of crashes"**.
- Placeholder windows, action strips, and chosen episodes unchanged
  (feng seed_0 M2: perceptual spawn (-6,0) 155 steps, kinematic pocket (-7,5)
  127 steps). Caption "The problem was never the driving. It was the seeing."
- Footage advice added to `MATERIALE_VIDEO/scena_12/README_capture.md`:
  - **Left / Perceptual** — spawn (-6,0): steers into wall, open space beside it.
    Capture **top-down (orthographic)** so the ignored gap is visible. End on impact.
    DaVinci: optional arrow pointing at the open space ("it was the seeing").
  - **Right / Kinematic** — spawn (-7,5): enters R_min pocket, can't rotate out,
    crashes. Capture **top-down** so the failed turn reads.
  - Both from the single existing command `./start_test_gui.sh 2 0 feng_hw_A 1 1`;
    speed-match each clip to its strip length (155 / 127 steps).

## Testing

- `tests/test_scene12_data.py` stays green (data contract unchanged).
- Where a script introduces a new numeric artist driven by data (results_table
  bars), add/keep an assertion or end-print confirming the rendered values equal
  the `scene12_data` means (the existing fidelity print covers this).
- Each script must run headless (Agg) and write its `.mp4` + `_end.png` to
  `MATERIALE_VIDEO/scena_12/`. Smoke check: render, confirm the end PNG visually.

## Out of scope

- Task 7 montage (`build_scene12_montage.py`) — still gated on user visual review.
- Gazebo capture / DaVinci compositing — manual, user-side.
- Audio edits — audio is frozen.

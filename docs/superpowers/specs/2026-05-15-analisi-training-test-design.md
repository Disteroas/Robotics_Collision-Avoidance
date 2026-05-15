# Analysis Script Design — `analisi.py`

## Goal

Single portable Python script for post-training analysis. Placed alongside the CSVs in a results folder. Generates clear plots and an AI-readable text summary.

## Architecture

Single file (`analisi.py`). No CLI args — paths relative to `Path(__file__).parent`. Two independent sections: Training and Test. Each section degrades gracefully if data is missing or columns are absent.

```
<results_folder>/
  analisi.py
  training_log.csv
  test_results.csv
  plots/
    01_reward_curve.png
    02_spawn_analysis.png
    03_crash_rate.png
    04_test_M1.png
    05_test_M2.png
    06_test_M3.png
  summary_training.txt
```

## Constants (top of file)

```python
MAX_STEPS    = 500    # max steps per episode — used to detect timeout
WINDOW       = 100    # rolling average window
MAZES_TESTED = [1, 2, 3]
```

---

## Input CSVs

### `training_log.csv`

Columns (merge14_05+): `ep_global, maze, steps, reward, avg100, epsilon, avg_loss, crashed, total_steps, total_crashes, spawn`

- `spawn`: string label e.g. `"(-6.0,0.0)"` — absent in older runs
- `crashed`: 1 if collision, 0 if timeout (reached MAX_STEPS)

### `test_results.csv`

Columns (base): `maze_id, episode, steps, reward, crashed, min_lidar, avg_lidar`

- Column `spawn` absent until `test.py` is patched (see §Prerequisite)
- `crashed`: 1 if collision, 0 if MAX_STEPS reached (success in this project)

---

## Prerequisite — patch `test.py`

`test.py` must be modified to log `spawn` per episode (same pattern as `train.py`).

Changes to `src/my_usv/scripts/test.py`:

1. CSV header: add `'spawn'` as last column
2. After `env.reset_environment(...)`: read `env.last_spawn` → format as `"({x:.1f},{y:.1f})"`
3. `csv_w.writerow(...)`: append spawn label as last element

This is required for per-spawn test analysis. The analysis script handles missing `spawn` column gracefully (global metrics only).

---

## Section 1 — Training

### Error handling

| Condition | Behavior |
|---|---|
| `training_log.csv` missing | `sys.exit("ERROR: training_log.csv not found")` |
| CSV empty (0 data rows) | `sys.exit("ERROR: training_log.csv is empty")` |
| `spawn` column missing | skip Plot 02, print `"WARNING: no spawn column — skipping spawn plots"` |
| No episode reaches MAX_STEPS | spawn max_steps subplot shows all 0.0, no error |

### Plot 01 — `01_reward_curve.png`

Single figure, one axis.

- **Thin semi-transparent line** (alpha=0.3): raw `reward` per episode
- **Thick solid line**: `reward.rolling(WINDOW, min_periods=1).mean()`
- X: `ep_global`. Y: reward (no clipping).
- Title: `"Training — Reward curve (rolling avg window={WINDOW})"` + folder name in subtitle
- Legend: `"reward (raw)"`, `"avg{WINDOW}"`
- Vertical dashed line where `avg100` first exceeds 0 (marks when training turns positive) — omit if never exceeds 0

### Plot 02 — `02_spawn_analysis.png`

Two subplots, stacked vertically. Only generated if `spawn` column present.

**Shared structure:**
- One bar per unique spawn point
- Bars sorted descending by subplot-1 value (avg steps)
- Same x-axis order in both subplots (sorted by avg steps, top to bottom)
- X tick labels: spawn label strings, rotated 45°

**Subplot 1 (top): Average steps per spawn point**
- `df.groupby("spawn")["steps"].mean()`
- Y: steps [0, MAX_STEPS]
- Horizontal dashed line at `MAX_STEPS` for reference
- Bar color: single color (steel blue)
- Error bars: ± std dev

**Subplot 2 (bottom): Max-steps rate per spawn point**
- `rate = (steps == MAX_STEPS).groupby(spawn) → sum / count`
- Y: fraction [0.0, 1.0], displayed as percentage on axis
- Interpretation: fraction of episodes that timed out (reached MAX_STEPS without crash) = survived
- Bar color: green (high) → red (low) colormap per bar value
- Add value labels on bars (`0.30` etc.)

Figure title: `"Spawn analysis — training M2"`

### Plot 03 — `03_crash_rate.png`

Single figure, one axis.

- Rolling crash rate: `crashed.rolling(WINDOW, min_periods=1).mean() * 100`
- Y: percentage [0, 100], labeled as `"Crash rate (%)" `
- X: `ep_global`
- Horizontal dashed line at 50% for reference
- Color: orange
- Title: `"Training — Rolling crash rate (window={WINDOW})"`

---

## Section 2 — Test

### Error handling

| Condition | Behavior |
|---|---|
| `test_results.csv` missing | print `"WARNING: test_results.csv not found — skipping test section"`, continue |
| Maze not in CSV | skip that maze's figure silently |
| `spawn` column missing | show global bars only, add note in plot subtitle |

### Plots 04–06 — `04_test_M{n}.png` (one per maze)

One figure per maze in `MAZES_TESTED`. Each figure has **2 subplots side by side**.

**Subplot left: Success rate**
- Success = `crashed == 0` (reached MAX_STEPS without collision)
- First bar: global rate (all episodes for this maze) — darker color, labeled "Global"
- Remaining bars: rate per unique spawn point — lighter color, labeled with spawn string
- Y: [0.0, 1.0] displayed as percentage
- Value labels on bars

**Subplot right: Avg steps per episode**
- First bar: global avg — darker color
- Remaining bars: per spawn point — lighter color
- Y: [0, MAX_STEPS]
- Horizontal dashed line at MAX_STEPS for reference
- Error bars: ± std on per-spawn bars

Figure title: `"Test results — Maze {n} ({n_ep} episodes)"`
Subtitle (if no spawn): `"spawn column missing — per-spawn breakdown unavailable"`

---

## `summary_training.txt`

AI-readable. ~30 lines. Structure:

```
=== USV DDQN — TRAINING SUMMARY ===
Generated: <datetime>
Source:    <folder name>/training_log.csv

--- CONFIGURATION ---
MAX_STEPS : 500   (episode ends at collision OR reaching this step count)
WINDOW    : 100   (rolling average window)
MAZE      : 2 only (M2)

--- COLUMN GUIDE ---
steps   : duration of episode in simulation steps
reward  : cumulative reward (positive if survived long, -1000 if crashed)
crashed : 1 = collision (bad), 0 = reached MAX_STEPS (good/timeout)
spawn   : spawn point label "(x,y)"

--- TRAINING OVERVIEW ---
Total episodes    : 4000
Total steps       : <value>
Epsilon (final)   : <value>
Crash rate (all)  : <value>%
Crash rate (last 100 ep) : <value>%

--- REWARD ---
Final avg{WINDOW}         : <value>
Best avg{WINDOW}          : <value>  (at ep <n>)
Avg reward (last 100 ep)  : <value>

--- SPAWN BREAKDOWN (training) ---
[if spawn column present]
Spawn point | Uses | Avg steps | Max-steps rate
(-6.0,0.0)  |  412 |    312.4  |  0.23
...
[sorted by avg steps desc]
[if spawn column absent]
"spawn column not present in CSV"

=== END ===
```

---

## Style conventions

- Font: default matplotlib, size 11 title / 9 tick labels
- DPI: 150
- Figure size: single-axis plots → (10, 4); 2-subplot vertical → (10, 7); 2-subplot horizontal (test) → (12, 5)
- Tight layout: always
- Save format: PNG
- Palette: consistent across script — blue for steps/reward, orange for crash rate, green/red colormap for rates

---

## Scope

This spec covers training section + test section plots and summary. Test section assumes `test.py` patched to log `spawn`. Script does not modify any CSV or checkpoint — read-only.

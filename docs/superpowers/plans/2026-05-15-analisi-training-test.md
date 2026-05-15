# Analysis Script Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `analysis/analisi_multimaze.py` — a portable analysis script that generates clear plots and an AI-readable text summary from `training_log.csv` and `test_results.csv`.

**Architecture:** Single Python file placed alongside CSVs. Two independent sections (Training, Test). Each section degrades gracefully when data/columns are missing. No CLI args — paths resolved via `Path(__file__).parent`. Also patches `test.py` to log spawn point per episode.

**Tech Stack:** Python 3, pandas, matplotlib, numpy, pathlib

---

## File Structure

| Action | Path | Purpose |
|---|---|---|
| Create | `analysis/analisi_multimaze.py` | Portable analysis script — copy to results folder to run |
| Modify | `src/my_usv/scripts/test.py:59-62,80,114-118` | Add `spawn` column to test_results.csv |

---

### Task 1: Patch test.py — add spawn column

**Context:** `test.py` already reads `env.last_spawn` is available after `reset_environment()` (same attribute added in merge14_05 for `train.py`). We need to add `spawn` to the CSV header and write it per episode.

**Files:**
- Modify: `src/my_usv/scripts/test.py`

- [ ] **Step 1: Update CSV header — add 'spawn' as last column**

In `src/my_usv/scripts/test.py`, find lines 58-62:
```python
    if csv_is_new:
        csv_w.writerow([
            'maze_id', 'episode', 'steps', 'reward', 'crashed',
            'min_lidar', 'avg_lidar'
        ])
```
Replace with:
```python
    if csv_is_new:
        csv_w.writerow([
            'maze_id', 'episode', 'steps', 'reward', 'crashed',
            'min_lidar', 'avg_lidar', 'spawn'
        ])
```

- [ ] **Step 2: Initialize spawn_label before the episode loop**

Find line 77 (`avg_lidars = []`). Add one line after it:
```python
    avg_lidars = []
    spawn_label = '?'   # updated each episode after reset
```

- [ ] **Step 3: Read spawn after reset_environment**

Find line 80:
```python
        state        = env.reset_environment(maze_id=args.maze_id, test_mode=True)
```
Replace with:
```python
        state        = env.reset_environment(maze_id=args.maze_id, test_mode=True)
        sx, sy, _    = env.last_spawn
        spawn_label  = f"({sx:.1f},{sy:.1f})"
```

- [ ] **Step 4: Add spawn_label to writerow**

Find lines 114-118:
```python
        csv_w.writerow([
            args.maze_id, ep, ep_steps, round(ep_reward, 2),
            int(crashed),
            round(min_lidars[-1], 3), round(avg_lidars[-1], 3)
        ])
```
Replace with:
```python
        csv_w.writerow([
            args.maze_id, ep, ep_steps, round(ep_reward, 2),
            int(crashed),
            round(min_lidars[-1], 3), round(avg_lidars[-1], 3),
            spawn_label
        ])
```

- [ ] **Step 5: Verify the patch is syntactically correct**

```bash
python3 -c "import ast; ast.parse(open('src/my_usv/scripts/test.py').read()); print('OK')"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add src/my_usv/scripts/test.py
git commit -m "feat(test): log spawn point per episode in test_results.csv"
```

---

### Task 2: Script skeleton — constants, imports, paths, CSV loading, main()

**Context:** The script lives in `analysis/` in the repo and is copied by the user alongside CSVs when running analysis. `Path(__file__).parent` resolves to whichever folder it's in at runtime.

**Files:**
- Create: `analysis/analisi_multimaze.py`

- [ ] **Step 1: Create the file with all boilerplate**

Create `analysis/analisi_multimaze.py`:

```python
"""
analisi_multimaze.py — Post-training analysis for USV DDQN multi-maze runs.

USAGE:
    Copy this script into a results folder alongside:
        training_log.csv   (required)
        test_results.csv   (optional — for test section)
    Then run:
        python analisi_multimaze.py

OUTPUT:
    plots/01_reward_curve.png
    plots/02_spawn_analysis.png   (only if 'spawn' column present)
    plots/03_crash_rate.png
    plots/04_test_M1.png          (only if test_results.csv present)
    plots/05_test_M2.png
    plots/06_test_M3.png
    summary_training.txt
"""

import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants — adjust to match the training run ──────────────────────────────
MAX_STEPS    = 500    # episode ends at collision OR this step count
WINDOW       = 100    # rolling average window (episodes)
MAZES_TESTED = [1, 2, 3]

# ── Paths — resolved relative to this script's location ───────────────────────
BASE_DIR    = Path(__file__).parent
TRAIN_CSV   = BASE_DIR / 'training_log.csv'
TEST_CSV    = BASE_DIR / 'test_results.csv'
PLOTS_DIR   = BASE_DIR / 'plots'
SUMMARY_TXT = BASE_DIR / 'summary_training.txt'


# ── Helpers ───────────────────────────────────────────────────────────────────

def setup_dirs():
    PLOTS_DIR.mkdir(exist_ok=True)


def load_training_csv():
    if not TRAIN_CSV.exists():
        sys.exit(f"ERROR: {TRAIN_CSV} not found. Place this script alongside training_log.csv.")
    df = pd.read_csv(TRAIN_CSV)
    if len(df) == 0:
        sys.exit(f"ERROR: {TRAIN_CSV} is empty.")
    return df


def load_test_csv():
    if not TEST_CSV.exists():
        print("WARNING: test_results.csv not found — skipping test section.")
        return None
    df = pd.read_csv(TEST_CSV)
    if len(df) == 0:
        print("WARNING: test_results.csv is empty — skipping test section.")
        return None
    return df


def save_fig(fig, filename):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ── Training plots ─────────────────────────────────────────────────────────────

def plot_reward_curve(df):
    pass  # Task 3


def plot_spawn_analysis(df):
    pass  # Task 4


def plot_crash_rate(df):
    pass  # Task 5


# ── Training summary ───────────────────────────────────────────────────────────

def write_summary(df):
    pass  # Task 6


# ── Test plots ─────────────────────────────────────────────────────────────────

def plot_test_maze(df_maze, maze_id):
    pass  # Task 7


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    setup_dirs()

    # ── Training section ──────────────────────────────────────────────────────
    print("\n[Training]")
    df_train = load_training_csv()

    plot_reward_curve(df_train)

    if 'spawn' not in df_train.columns:
        print("  WARNING: no 'spawn' column — skipping spawn plots.")
    else:
        plot_spawn_analysis(df_train)

    plot_crash_rate(df_train)
    write_summary(df_train)

    # ── Test section ──────────────────────────────────────────────────────────
    print("\n[Test]")
    df_test = load_test_csv()
    if df_test is not None:
        for maze_id in MAZES_TESTED:
            df_maze = df_test[df_test['maze_id'] == maze_id]
            if len(df_maze) == 0:
                continue
            plot_test_maze(df_maze, maze_id)

    print(f"\nDone. Output in: {BASE_DIR}\n")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify skeleton runs without error on a minimal CSV**

```bash
# create minimal training_log.csv in a temp folder
python3 -c "
import pandas as pd, pathlib
p = pathlib.Path('/tmp/test_analisi')
p.mkdir(exist_ok=True)
pd.DataFrame({
    'ep_global': range(1,11), 'maze': [2]*10,
    'steps': [50]*10, 'reward': [-100.0]*10,
    'avg100': [-100.0]*10, 'epsilon': [0.9]*10,
    'avg_loss': [100.0]*10, 'crashed': [1]*10,
    'total_steps': list(range(50,550,50)),
    'total_crashes': list(range(1,11))
}).to_csv(p / 'training_log.csv', index=False)
"
cp analysis/analisi_multimaze.py /tmp/test_analisi/
cd /tmp/test_analisi && python3 analisi_multimaze.py
```
Expected output:
```
[Training]
  WARNING: no 'spawn' column — skipping spawn plots.

[Test]
WARNING: test_results.csv not found — skipping test section.

Done. Output in: /tmp/test_analisi
```
No tracebacks. Directory `plots/` created.

- [ ] **Step 3: Commit skeleton**

```bash
git add analysis/analisi_multimaze.py
git commit -m "feat(analysis): analisi_multimaze.py skeleton — constants, loading, main"
```

---

### Task 3: Plot 01 — reward curve

**Files:**
- Modify: `analysis/analisi_multimaze.py` — implement `plot_reward_curve(df)`

- [ ] **Step 1: Implement plot_reward_curve**

Replace the `pass` in `plot_reward_curve(df)` with:

```python
def plot_reward_curve(df):
    ep      = df['ep_global']
    reward  = df['reward']
    rolling = reward.rolling(WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(ep, reward,  alpha=0.25, lw=0.8, color='steelblue', label='reward (raw)')
    ax.plot(ep, rolling, lw=2.0,     color='steelblue',          label=f'avg{WINDOW}')

    # Mark where avg first turns positive
    pos = rolling[rolling > 0]
    if len(pos) > 0:
        first_pos_ep = int(ep.iloc[pos.index[0]])
        ax.axvline(first_pos_ep, color='green', linestyle='--', alpha=0.7,
                   label=f'avg > 0 (ep {first_pos_ep})')

    ax.axhline(0, color='black', lw=0.6, ls='--', alpha=0.4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title(f'Training — Reward curve  (rolling avg window={WINDOW})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, '01_reward_curve.png')
```

- [ ] **Step 2: Run and verify output**

```bash
cd /tmp/test_analisi && python3 analisi_multimaze.py
```
Expected: `Saved: 01_reward_curve.png` in output. File `plots/01_reward_curve.png` exists.

```bash
ls /tmp/test_analisi/plots/
```
Expected: `01_reward_curve.png`

- [ ] **Step 3: Commit**

```bash
git add analysis/analisi_multimaze.py
git commit -m "feat(analysis): plot 01 — reward curve with rolling avg"
```

---

### Task 4: Plot 02 — spawn analysis (training)

**Files:**
- Modify: `analysis/analisi_multimaze.py` — implement `plot_spawn_analysis(df)`

- [ ] **Step 1: Implement plot_spawn_analysis**

Replace the `pass` in `plot_spawn_analysis(df)` with:

```python
def plot_spawn_analysis(df):
    grp       = df.groupby('spawn')
    avg_steps = grp['steps'].mean().sort_values(ascending=False)
    std_steps = grp['steps'].std().reindex(avg_steps.index).fillna(0)
    max_rate  = (df['steps'] == MAX_STEPS).groupby(df['spawn']).mean().reindex(avg_steps.index)
    uses      = grp.size().reindex(avg_steps.index)

    labels = avg_steps.index.tolist()
    x      = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # ── Subplot 1: avg steps ──────────────────────────────────────────────────
    ax1.bar(x, avg_steps.values, color='steelblue', alpha=0.85)
    ax1.errorbar(x, avg_steps.values, yerr=std_steps.values,
                 fmt='none', color='black', capsize=4, linewidth=1)
    ax1.axhline(MAX_STEPS, color='gray', linestyle='--', alpha=0.6,
                label=f'MAX_STEPS={MAX_STEPS}')
    ax1.set_ylabel('Avg steps')
    ax1.set_title('Avg steps per spawn point  (higher = survives longer)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, MAX_STEPS * 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # annotate n uses
    for xi, (label, n) in enumerate(zip(labels, uses.values)):
        ax1.text(xi, avg_steps.values[xi] + MAX_STEPS * 0.01,
                 f'n={n}', ha='center', va='bottom', fontsize=7, color='gray')

    # ── Subplot 2: max-steps rate ─────────────────────────────────────────────
    rates  = max_rate.values
    colors = [plt.cm.RdYlGn(float(v)) for v in rates]
    bars   = ax2.bar(x, rates, color=colors, alpha=0.9)
    for bar, val in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Max-steps rate')
    ax2.set_title('Fraction of episodes reaching MAX_STEPS without crash  (higher = better)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.18)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Spawn analysis — Training', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '02_spawn_analysis.png')
```

- [ ] **Step 2: Verify with CSV that has spawn column**

```bash
python3 -c "
import pandas as pd, pathlib
p = pathlib.Path('/tmp/test_analisi')
pd.DataFrame({
    'ep_global': range(1, 21), 'maze': [2]*20,
    'steps': [50,500,50,300,50,500,50,100,50,400]*2,
    'reward': [-100.0]*20, 'avg100': [-100.0]*20,
    'epsilon': [0.9]*20, 'avg_loss': [100.0]*20,
    'crashed': [1,0,1,1,1,0,1,1,1,1]*2,
    'total_steps': list(range(50,1050,50)),
    'total_crashes': list(range(1,21)),
    'spawn': ['(-6.0,0.0)']*10 + ['(-2.0,1.0)']*10
}).to_csv(p / 'training_log.csv', index=False)
"
cd /tmp/test_analisi && python3 analisi_multimaze.py
```
Expected: `Saved: 02_spawn_analysis.png` in output.

- [ ] **Step 3: Commit**

```bash
git add analysis/analisi_multimaze.py
git commit -m "feat(analysis): plot 02 — spawn analysis (avg steps + max-steps rate)"
```

---

### Task 5: Plot 03 — crash rate (training)

**Files:**
- Modify: `analysis/analisi_multimaze.py` — implement `plot_crash_rate(df)`

- [ ] **Step 1: Implement plot_crash_rate**

Replace the `pass` in `plot_crash_rate(df)` with:

```python
def plot_crash_rate(df):
    roll_crash = df['crashed'].rolling(WINDOW, min_periods=1).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(df['ep_global'], roll_crash, lw=1.8, color='darkorange')
    ax.fill_between(df['ep_global'], roll_crash, alpha=0.12, color='darkorange')
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax.axhline(0,  color='green', linestyle='--', alpha=0.4, label='0%')

    ax.set_xlabel('Episode')
    ax.set_ylabel('Crash rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title(f'Training — Rolling crash rate  (window={WINDOW})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_fig(fig, '03_crash_rate.png')
```

- [ ] **Step 2: Verify output file created**

```bash
cd /tmp/test_analisi && python3 analisi_multimaze.py
ls /tmp/test_analisi/plots/
```
Expected: `01_reward_curve.png  02_spawn_analysis.png  03_crash_rate.png`

- [ ] **Step 3: Commit**

```bash
git add analysis/analisi_multimaze.py
git commit -m "feat(analysis): plot 03 — rolling crash rate"
```

---

### Task 6: summary_training.txt

**Files:**
- Modify: `analysis/analisi_multimaze.py` — implement `write_summary(df)`

- [ ] **Step 1: Implement write_summary**

Replace the `pass` in `write_summary(df)` with:

```python
def write_summary(df):
    has_spawn   = 'spawn' in df.columns
    total_ep    = len(df)
    total_steps = int(df['total_steps'].iloc[-1]) if 'total_steps' in df.columns else int(df['steps'].sum())
    eps_final   = round(float(df['epsilon'].iloc[-1]), 4)
    crash_all   = round(df['crashed'].mean() * 100, 1)
    crash_last  = round(df['crashed'].tail(WINDOW).mean() * 100, 1)
    rolling     = df['reward'].rolling(WINDOW, min_periods=1).mean()
    avg_last    = round(float(rolling.iloc[-1]), 1)
    best_avg    = round(float(rolling.max()), 1)
    best_ep     = int(df['ep_global'].iloc[rolling.idxmax()])
    maze_str    = ', '.join(str(m) for m in sorted(df['maze'].unique()))

    lines = [
        "=== USV DDQN — TRAINING SUMMARY ===",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source    : {BASE_DIR.name}/training_log.csv",
        "",
        "--- CONFIGURATION ---",
        f"MAX_STEPS : {MAX_STEPS}   (episode ends at collision OR reaching this step count)",
        f"WINDOW    : {WINDOW}   (rolling average window for metrics below)",
        f"MAZE(S)   : {maze_str}",
        "",
        "--- COLUMN GUIDE ---",
        "steps   : duration of episode in simulation steps (1 to MAX_STEPS)",
        "reward  : cumulative reward (higher = better; approx -1000 if crashed early)",
        "crashed : 1 = collision (bad), 0 = reached MAX_STEPS without crash (good)",
        "spawn   : spawn point label '(x,y)' — absent in runs before merge14_05",
        "",
        "--- TRAINING OVERVIEW ---",
        f"Total episodes          : {total_ep}",
        f"Total steps             : {total_steps}",
        f"Epsilon (final)         : {eps_final}",
        f"Crash rate (all ep)     : {crash_all}%",
        f"Crash rate (last {WINDOW} ep) : {crash_last}%",
        "",
        "--- REWARD ---",
        f"Final avg{WINDOW}              : {avg_last}  (rolling avg over last {WINDOW} ep)",
        f"Best avg{WINDOW}               : {best_avg}  (at ep {best_ep})",
        "",
    ]

    if has_spawn:
        grp       = df.groupby('spawn')
        avg_steps = grp['steps'].mean().sort_values(ascending=False)
        max_rate  = (df['steps'] == MAX_STEPS).groupby(df['spawn']).mean()
        uses      = grp.size()

        lines.append("--- SPAWN BREAKDOWN (training) ---")
        lines.append("Sorted by avg_steps descending. max_steps_rate = fraction of episodes")
        lines.append("that reached MAX_STEPS without crashing (higher = spawn point better handled).")
        lines.append("")
        lines.append(f"{'Spawn':<18} {'Uses':>5} {'Avg steps':>10} {'Max-steps rate':>15}")
        lines.append("-" * 52)
        for spawn in avg_steps.index:
            lines.append(
                f"{spawn:<18} {uses[spawn]:>5} {avg_steps[spawn]:>10.1f} "
                f"{max_rate.get(spawn, 0.0):>14.1%}"
            )
        lines.append("")
    else:
        lines.append("--- SPAWN BREAKDOWN ---")
        lines.append("spawn column not present in CSV (run predates merge14_05).")
        lines.append("")

    lines.append("=== END ===")

    SUMMARY_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Saved: summary_training.txt")
```

- [ ] **Step 2: Verify summary file content**

```bash
cd /tmp/test_analisi && python3 analisi_multimaze.py && cat summary_training.txt
```
Expected: structured text with all sections, no tracebacks. Spawn table present (CSV has spawn column).

- [ ] **Step 3: Commit**

```bash
git add analysis/analisi_multimaze.py
git commit -m "feat(analysis): summary_training.txt — AI-readable training report"
```

---

### Task 7: Test section — plot per maze

**Files:**
- Modify: `analysis/analisi_multimaze.py` — implement `plot_test_maze(df_maze, maze_id)`

- [ ] **Step 1: Implement plot_test_maze**

Replace the `pass` in `plot_test_maze(df_maze, maze_id)` with:

```python
def plot_test_maze(df_maze, maze_id):
    n_ep      = len(df_maze)
    has_spawn = 'spawn' in df_maze.columns

    # Global metrics
    global_success    = 1.0 - df_maze['crashed'].mean()
    global_avg_steps  = df_maze['steps'].mean()
    global_std_steps  = df_maze['steps'].std()

    if has_spawn:
        grp             = df_maze.groupby('spawn')
        spawn_success   = 1.0 - grp['crashed'].mean()
        spawn_avg_steps = grp['steps'].mean()
        spawn_std_steps = grp['steps'].std().fillna(0)

        # Sort spawns by success rate descending for readability
        order = spawn_success.sort_values(ascending=False).index.tolist()

        labels      = ['Global'] + order
        succ_vals   = [global_success]   + [spawn_success[s]   for s in order]
        steps_vals  = [global_avg_steps] + [spawn_avg_steps[s] for s in order]
        steps_errs  = [global_std_steps] + [spawn_std_steps[s] for s in order]
    else:
        labels     = ['Global']
        succ_vals  = [global_success]
        steps_vals = [global_avg_steps]
        steps_errs = [global_std_steps]

    x      = np.arange(len(labels))
    colors = ['#2c7bb6'] + ['#74add1'] * (len(labels) - 1)  # Global darker

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left: success rate ────────────────────────────────────────────────────
    bars1 = ax1.bar(x, succ_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars1, succ_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('Success rate')
    ax1.set_title('Success rate  (crashed == 0)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax1.grid(True, alpha=0.3, axis='y')

    # ── Right: avg steps ──────────────────────────────────────────────────────
    ax2.bar(x, steps_vals, color=colors, alpha=0.85)
    ax2.errorbar(x, steps_vals, yerr=steps_errs,
                 fmt='none', color='black', capsize=4, linewidth=1)
    ax2.axhline(MAX_STEPS, color='green', linestyle='--', alpha=0.5,
                label=f'MAX_STEPS={MAX_STEPS}')
    ax2.set_ylabel('Avg steps')
    ax2.set_title('Avg steps per episode')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, MAX_STEPS * 1.25)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    subtitle = '' if has_spawn else '\n(spawn column missing — per-spawn breakdown unavailable)'
    fig.suptitle(f'Test results — Maze {maze_id}  ({n_ep} episodes){subtitle}',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()

    fname = f'{3 + maze_id:02d}_test_M{maze_id}.png'   # 04, 05, 06
    save_fig(fig, fname)
```

- [ ] **Step 2: Create minimal test CSV and verify**

```bash
python3 -c "
import pandas as pd, pathlib
p = pathlib.Path('/tmp/test_analisi')
rows = []
spawns = ['(-6.0,0.0)','(-2.0,1.0)','(-4.0,2.0)']
for maze in [1,2,3]:
    for i in range(9):
        rows.append({
            'maze_id': maze, 'episode': i+1,
            'steps': 500 if i%3==0 else 50,
            'reward': 2500.0 if i%3==0 else -100.0,
            'crashed': 0 if i%3==0 else 1,
            'min_lidar': 0.5, 'avg_lidar': 1.2,
            'spawn': spawns[i%3]
        })
pd.DataFrame(rows).to_csv(p / 'test_results.csv', index=False)
"
cd /tmp/test_analisi && python3 analisi_multimaze.py
ls /tmp/test_analisi/plots/
```
Expected output includes:
```
  Saved: 04_test_M1.png
  Saved: 05_test_M2.png
  Saved: 06_test_M3.png
```
Expected files: `01_reward_curve.png  02_spawn_analysis.png  03_crash_rate.png  04_test_M1.png  05_test_M2.png  06_test_M3.png`

- [ ] **Step 3: Verify graceful degradation — test CSV without spawn column**

```bash
python3 -c "
import pandas as pd, pathlib
p = pathlib.Path('/tmp/test_analisi')
pd.DataFrame({
    'maze_id': [2]*5, 'episode': range(1,6),
    'steps': [500,50,300,50,500],
    'reward': [2500,-100,100,-100,2500],
    'crashed': [0,1,1,1,0],
    'min_lidar': [0.5]*5, 'avg_lidar': [1.2]*5
}).to_csv(p / 'test_results.csv', index=False)
"
cd /tmp/test_analisi && python3 analisi_multimaze.py 2>&1 | grep -E "Saved|WARNING|ERROR"
```
Expected: `Saved: 05_test_M2.png` with subtitle note in the figure (no crash).

- [ ] **Step 4: Commit**

```bash
git add analysis/analisi_multimaze.py
git commit -m "feat(analysis): test section — per-maze plots (success rate + avg steps)"
```

---

### Task 8: Final integration commit + push

**Files:**
- No new changes — final verification and push

- [ ] **Step 1: Run full smoke test with both CSVs (spawn present)**

Use the synthetic data from Tasks 4 and 7 (both CSVs in `/tmp/test_analisi/` with spawn column).

```bash
cd /tmp/test_analisi && python3 analisi_multimaze.py
```
Expected full output (no tracebacks):
```
[Training]
  Saved: 01_reward_curve.png
  Saved: 02_spawn_analysis.png
  Saved: 03_crash_rate.png
  Saved: summary_training.txt

[Test]
  Saved: 04_test_M1.png
  Saved: 05_test_M2.png
  Saved: 06_test_M3.png

Done. Output in: /tmp/test_analisi
```

- [ ] **Step 2: Run with missing training CSV — verify exit message**

```bash
cd /tmp && mkdir test_empty && cp test_analisi/analisi_multimaze.py test_empty/
cd /tmp/test_empty && python3 analisi_multimaze.py 2>&1
```
Expected: `ERROR: .../training_log.csv not found. Place this script alongside training_log.csv.`
No traceback.

- [ ] **Step 3: Final commit and push**

```bash
git add analysis/analisi_multimaze.py
git push
```

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
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants — adjust to match the training run ──────────────────────────────
MAX_STEPS    = 500    # episode ends at collision OR this step count
WINDOW       = 100    # rolling average window (episodes)
MAZES_TESTED = (1, 2, 3)

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
    REQUIRED = {'ep_global', 'maze', 'steps', 'reward', 'crashed', 'epsilon'}
    missing  = REQUIRED - set(df.columns)
    if missing:
        sys.exit(f"ERROR: {TRAIN_CSV} missing required columns: {missing}")
    if df.empty:
        sys.exit(f"ERROR: {TRAIN_CSV} is empty.")
    return df


def load_test_csv():
    if not TEST_CSV.exists():
        print("WARNING: test_results.csv not found — skipping test section.")
        return None
    df = pd.read_csv(TEST_CSV)
    if df.empty:
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
    ep      = df['ep_global']
    reward  = df['reward']
    rolling = reward.rolling(WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(ep, reward,  alpha=0.25, lw=0.8, color='steelblue', label='reward (raw)')
    ax.plot(ep, rolling, lw=2.0,     color='steelblue',          label=f'avg{WINDOW}')

    # Mark where avg first turns positive
    pos = rolling[rolling > 0]
    if len(pos) > 0:
        first_pos_ep = int(ep.loc[pos.index[0]])
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


def plot_spawn_analysis(df):
    grp       = df.groupby('spawn')
    avg_steps = grp['steps'].mean().sort_values(ascending=False)
    std_steps = grp['steps'].std().reindex(avg_steps.index).fillna(0)
    max_rate  = (df['steps'] == MAX_STEPS).groupby(df['spawn']).mean().reindex(avg_steps.index).fillna(0)
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
                print(f"  WARNING: no rows for maze {maze_id} in test CSV — skipping.")
                continue
            plot_test_maze(df_maze, maze_id)

    print(f"\nDone. Output in: {BASE_DIR}\n")


if __name__ == '__main__':
    main()

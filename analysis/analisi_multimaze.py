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
                print(f"  WARNING: no rows for maze {maze_id} in test CSV — skipping.")
                continue
            plot_test_maze(df_maze, maze_id)

    print(f"\nDone. Output in: {BASE_DIR}\n")


if __name__ == '__main__':
    main()

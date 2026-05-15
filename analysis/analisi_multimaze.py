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

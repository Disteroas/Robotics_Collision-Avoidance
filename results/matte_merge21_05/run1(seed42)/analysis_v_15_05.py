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
    plots/04_loss_curve.png       (only if 'avg_loss' column present)
    plots/05_test_M1.png          (only if test_results.csv present)
    plots/06_test_M2.png
    plots/07_test_M3.png
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

    # FIX: Aggiunto .to_numpy()
    ax.plot(ep.to_numpy(), reward.to_numpy(),  alpha=0.25, lw=0.8, color='steelblue', label='reward (raw)')
    ax.plot(ep.to_numpy(), rolling.to_numpy(), lw=2.0,     color='steelblue',          label=f'avg{WINDOW}')

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
    roll_crash = df['crashed'].rolling(WINDOW, min_periods=1).mean() * 100

    fig, ax = plt.subplots(figsize=(10, 4))

    # FIX: Aggiunto .to_numpy()
    ax.plot(df['ep_global'].to_numpy(), roll_crash.to_numpy(), lw=1.8, color='darkorange')
    ax.fill_between(df['ep_global'].to_numpy(), roll_crash.to_numpy(), alpha=0.12, color='darkorange')
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


def plot_loss_curve(df):
    loss    = df['avg_loss']
    rolling = loss.rolling(WINDOW, min_periods=1).mean()
    ep      = df['ep_global']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # ── Linear scale ──────────────────────────────────────────────────────────
    # FIX: Aggiunto .to_numpy()
    ax1.plot(ep.to_numpy(), loss.to_numpy(),    alpha=0.2, lw=0.8, color='mediumpurple', label='loss (raw)')
    ax1.plot(ep.to_numpy(), rolling.to_numpy(), lw=2.0,    color='mediumpurple',          label=f'avg{WINDOW}')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'Training — Loss curve  (linear scale, window={WINDOW})')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ── Log scale — reveals explosive gradient growth ─────────────────────────
    mask     = loss > 0
    ep_pos   = ep[mask]
    loss_pos = loss[mask]
    roll_pos = rolling[mask].clip(lower=1e-6)

    if len(loss_pos) > 0:
        # FIX: Aggiunto .to_numpy()
        ax2.plot(ep_pos.to_numpy(), loss_pos.to_numpy(), alpha=0.2, lw=0.8, color='mediumpurple', label='loss (raw)')
        ax2.plot(ep_pos.to_numpy(), roll_pos.to_numpy(), lw=2.0,    color='mediumpurple',          label=f'avg{WINDOW}')
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss (log scale)')
        ax2.set_title('Log scale — stable convergence = flat; explosive = rapid rise')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which='both')
    else:
        ax2.text(0.5, 0.5, 'No positive loss values to display',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=10)

    ax2.set_xlabel('Episode')
    fig.suptitle('Training — Loss (MSE)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '04_loss_curve.png')


# ── Training summary ───────────────────────────────────────────────────────────

def write_summary(df):
    from datetime import datetime

    has_spawn   = 'spawn' in df.columns
    total_ep    = len(df)
    total_steps = int(df['total_steps'].iloc[-1]) if 'total_steps' in df.columns else int(df['steps'].sum())
    eps_final   = round(float(df['epsilon'].iloc[-1]), 4)
    crash_all   = round(df['crashed'].mean() * 100, 1)
    crash_last  = round(df['crashed'].tail(WINDOW).mean() * 100, 1)
    rolling     = df['reward'].rolling(WINDOW, min_periods=1).mean()
    avg_last    = round(float(rolling.iloc[-1]), 1)
    best_avg    = round(float(rolling.max()), 1)
    best_ep     = int(df['ep_global'].loc[rolling.idxmax()])
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
        max_rate  = (df['steps'] == MAX_STEPS).groupby(df['spawn']).mean().reindex(avg_steps.index).fillna(0)
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


# ── Test plots ─────────────────────────────────────────────────────────────────

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

    fname = f'{4 + maze_id:02d}_test_M{maze_id}.png'   # 05, 06, 07
    save_fig(fig, fname)


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

    if 'avg_loss' not in df_train.columns:
        print("  WARNING: no 'avg_loss' column — skipping loss plot.")
    else:
        plot_loss_curve(df_train)

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

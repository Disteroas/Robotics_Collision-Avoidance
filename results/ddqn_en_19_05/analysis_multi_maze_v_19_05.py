"""
analysis_multi_maze_v_19_05.py — Post-training analysis for ddqn_en_19_05 run.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_STEPS    = 500
WINDOW       = 100        # rolling avg window (global)
WINDOW_MAZE  = 30         # rolling avg window per-maze (more reactive: blocks ~100 ep)
MAZES_TRAIN  = (1, 2)
MAZES_TESTED = (1, 2, 3)

MAZE_COLORS  = {1: '#d62728', 2: '#1f77b4', 3: '#2ca02c'}  # red M1 / blue M2 / green M3
MAZE_LABEL   = {1: 'M1', 2: 'M2', 3: 'M3'}

# ── Paths ─────────────────────────────────────────────────────────────────────
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
        sys.exit(f"ERROR: {TRAIN_CSV} not found.")
    df = pd.read_csv(TRAIN_CSV)
    REQUIRED = {'ep_global', 'maze', 'steps', 'reward', 'crashed', 'epsilon'}
    missing  = REQUIRED - set(df.columns)
    if missing:
        sys.exit(f"ERROR: missing columns: {missing}")
    if df.empty:
        sys.exit(f"ERROR: empty CSV.")
    return df


def load_test_csv():
    if not TEST_CSV.exists():
        print("  test_results.csv mancante — skip test section.")
        return None
    df = pd.read_csv(TEST_CSV)
    return df if not df.empty else None


def save_fig(fig, filename):
    path = PLOTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {filename}")


# ── Plot 01: global reward curve + per-maze overlay ───────────────────────────

def plot_reward_global(df):
    ep      = df['ep_global']
    reward  = df['reward']
    rolling = reward.rolling(WINDOW, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(ep.to_numpy(), reward.to_numpy(), alpha=0.18, lw=0.6, color='gray', label='reward (raw)')
    ax.plot(ep.to_numpy(), rolling.to_numpy(), lw=2.0,    color='black', label=f'global avg{WINDOW}')

    # Overlay per-maze rolling mean (over ALL episodes, masked by maze)
    for m in MAZES_TRAIN:
        mask = df['maze'] == m
        if mask.sum() == 0:
            continue
        sub_ep   = df.loc[mask, 'ep_global']
        sub_rew  = df.loc[mask, 'reward']
        sub_roll = sub_rew.rolling(WINDOW_MAZE, min_periods=1).mean()
        ax.plot(sub_ep.to_numpy(), sub_roll.to_numpy(), lw=1.8, color=MAZE_COLORS[m], alpha=0.9,
                label=f'{MAZE_LABEL[m]} avg{WINDOW_MAZE}')

    pos = rolling[rolling > 0]
    if len(pos) > 0:
        first_pos_ep = int(ep.loc[pos.index[0]])
        ax.axvline(first_pos_ep, color='green', linestyle='--', alpha=0.5,
                   label=f'global avg>0 (ep {first_pos_ep})')

    ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Reward — global + per-maze rolling')
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, '01_reward_curve_global.png')


# ── Plot 02: reward per-maze (subplot per maze) ───────────────────────────────

def plot_reward_per_maze(df):
    fig, axes = plt.subplots(len(MAZES_TRAIN), 1, figsize=(11, 3.5 * len(MAZES_TRAIN)),
                             sharex=True)
    if len(MAZES_TRAIN) == 1:
        axes = [axes]

    for ax, m in zip(axes, MAZES_TRAIN):
        sub = df[df['maze'] == m]
        if sub.empty:
            ax.text(0.5, 0.5, f'No data for M{m}',
                    transform=ax.transAxes, ha='center', va='center')
            continue
        ep      = sub['ep_global']
        reward  = sub['reward']
        rolling = reward.rolling(WINDOW_MAZE, min_periods=1).mean()
        color   = MAZE_COLORS[m]

        ax.plot(ep.to_numpy(), reward.to_numpy(),  alpha=0.25, lw=0.7, color=color, label=f'{MAZE_LABEL[m]} raw')
        ax.plot(ep.to_numpy(), rolling.to_numpy(), lw=2.0,    color=color, label=f'{MAZE_LABEL[m]} avg{WINDOW_MAZE}')
        ax.axhline(0, color='black', lw=0.5, ls='--', alpha=0.4)
        ax.set_ylabel('Reward')
        ax.set_title(f'{MAZE_LABEL[m]} only — n={len(sub)} episodes')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Episode (global)')
    fig.suptitle('Reward per-maze (training blocks)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '02_reward_curve_per_maze.png')


# ── Plot 03: crash rate per-maze ──────────────────────────────────────────────

def plot_crash_per_maze(df):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    # Global
    roll_all = df['crashed'].rolling(WINDOW, min_periods=1).mean() * 100
    ax.plot(df['ep_global'].to_numpy(), roll_all.to_numpy(), lw=1.6, color='black', alpha=0.7,
            label=f'global avg{WINDOW}')

    for m in MAZES_TRAIN:
        sub = df[df['maze'] == m]
        if sub.empty:
            continue
        roll = sub['crashed'].rolling(WINDOW_MAZE, min_periods=1).mean() * 100
        ax.plot(sub['ep_global'].to_numpy(), roll.to_numpy(), lw=2.0, color=MAZE_COLORS[m], alpha=0.9,
                label=f'{MAZE_LABEL[m]} avg{WINDOW_MAZE}')

    ax.axhline(50, color='gray', linestyle='--', alpha=0.4)
    ax.axhline(0,  color='green', linestyle='--', alpha=0.4)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Crash rate (%)')
    ax.set_ylim(0, 105)
    ax.set_title('Crash rate per-maze')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, '03_crash_rate_per_maze.png')


# ── Plot 04: steps survived per-maze ──────────────────────────────────────────

def plot_steps_per_maze(df):
    fig, ax = plt.subplots(figsize=(11, 4.5))

    for m in MAZES_TRAIN:
        sub = df[df['maze'] == m]
        if sub.empty:
            continue
        roll = sub['steps'].rolling(WINDOW_MAZE, min_periods=1).mean()
        ax.plot(sub['ep_global'].to_numpy(), roll.to_numpy(), lw=2.0, color=MAZE_COLORS[m], alpha=0.9,
                label=f'{MAZE_LABEL[m]} avg{WINDOW_MAZE}')

    ax.axhline(MAX_STEPS, color='green', linestyle='--', alpha=0.5,
               label=f'MAX_STEPS={MAX_STEPS}')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Steps per episode')
    ax.set_ylim(0, MAX_STEPS * 1.1)
    ax.set_title('Steps survived per-maze (rolling)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, '04_steps_per_maze.png')


# ── Plot 05/06: spawn analysis per maze ───────────────────────────────────────

def plot_spawn_per_maze(df, maze_id):
    sub = df[df['maze'] == maze_id]
    if sub.empty or 'spawn' not in sub.columns:
        return

    grp       = sub.groupby('spawn')
    avg_steps = grp['steps'].mean().sort_values(ascending=False)
    std_steps = grp['steps'].std().reindex(avg_steps.index).fillna(0)
    max_rate  = (sub['steps'] == MAX_STEPS).groupby(sub['spawn']).mean().reindex(avg_steps.index).fillna(0)
    uses      = grp.size().reindex(avg_steps.index)

    labels = avg_steps.index.tolist()
    x      = np.arange(len(labels))
    color  = MAZE_COLORS[maze_id]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    ax1.bar(x, avg_steps.values, color=color, alpha=0.85)
    ax1.errorbar(x, avg_steps.values, yerr=std_steps.values,
                 fmt='none', color='black', capsize=4, linewidth=1)
    ax1.axhline(MAX_STEPS, color='gray', linestyle='--', alpha=0.6,
                label=f'MAX_STEPS={MAX_STEPS}')
    ax1.set_ylabel('Avg steps')
    ax1.set_title(f'{MAZE_LABEL[maze_id]} — avg steps per spawn')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, MAX_STEPS * 1.15)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    for xi, (label, n) in enumerate(zip(labels, uses.values)):
        ax1.text(xi, avg_steps.values[xi] + MAX_STEPS * 0.01,
                 f'n={n}', ha='center', va='bottom', fontsize=7, color='gray')

    rates  = max_rate.values
    colors = [plt.cm.RdYlGn(float(v)) for v in rates]
    bars   = ax2.bar(x, rates, color=colors, alpha=0.9)
    for bar, val in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    ax2.set_ylabel('Max-steps rate')
    ax2.set_title(f'{MAZE_LABEL[maze_id]} — fraction reaching MAX_STEPS without crash')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylim(0, 1.18)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax2.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Spawn analysis — {MAZE_LABEL[maze_id]} training',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname = {1: '05_spawn_analysis_M1.png', 2: '06_spawn_analysis_M2.png'}[maze_id]
    save_fig(fig, fname)


# ── Plot 07: loss curve ───────────────────────────────────────────────────────

def plot_loss_curve(df):
    if 'avg_loss' not in df.columns:
        print("  avg_loss column mancante — skip loss plot.")
        return

    loss    = df['avg_loss']
    rolling = loss.rolling(WINDOW, min_periods=1).mean()
    ep      = df['ep_global']

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

    ax1.plot(ep.to_numpy(), loss.to_numpy(),    alpha=0.2, lw=0.7, color='mediumpurple', label='loss (raw)')
    ax1.plot(ep.to_numpy(), rolling.to_numpy(), lw=2.0,    color='mediumpurple',        label=f'avg{WINDOW}')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title(f'Loss — linear scale (window={WINDOW})')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    mask     = loss > 0
    ep_pos   = ep[mask]
    loss_pos = loss[mask]
    roll_pos = rolling[mask].clip(lower=1e-6)
    if len(loss_pos) > 0:
        ax2.plot(ep_pos.to_numpy(), loss_pos.to_numpy(), alpha=0.2, lw=0.7, color='mediumpurple', label='loss (raw)')
        ax2.plot(ep_pos.to_numpy(), roll_pos.to_numpy(), lw=2.0,    color='mediumpurple',        label=f'avg{WINDOW}')
        ax2.set_yscale('log')
        ax2.set_ylabel('Loss (log)')
        ax2.set_title('Log scale — stable convergence flat, explosive rises rapid')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3, which='both')

    ax2.set_xlabel('Episode')
    fig.suptitle('Training — Loss (MSE)', fontsize=12, fontweight='bold')
    fig.tight_layout()
    save_fig(fig, '07_loss_curve.png')


# ── Plot 08: maze block timeline ──────────────────────────────────────────────

def plot_maze_timeline(df):
    fig, ax = plt.subplots(figsize=(11, 2.5))

    for m in MAZES_TRAIN:
        sub_ep = df.loc[df['maze'] == m, 'ep_global']
        if sub_ep.empty:
            continue
        ax.scatter(sub_ep.to_numpy(), [m] * len(sub_ep), s=4,
                   color=MAZE_COLORS[m], alpha=0.6, label=MAZE_LABEL[m])

    ax.set_xlabel('Episode')
    ax.set_ylabel('Maze ID')
    ax.set_yticks(list(MAZES_TRAIN))
    ax.set_yticklabels([MAZE_LABEL[m] for m in MAZES_TRAIN])
    ax.set_title('Maze block timeline (which maze trained per episode)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    save_fig(fig, '08_maze_timeline.png')


# ── Summary ───────────────────────────────────────────────────────────────────

def write_summary(df):
    from datetime import datetime

    has_spawn   = 'spawn' in df.columns
    total_ep    = len(df)
    total_steps = int(df['total_steps'].iloc[-1]) if 'total_steps' in df.columns else int(df['steps'].sum())
    eps_final   = round(float(df['epsilon'].iloc[-1]), 4)
    maze_str    = ', '.join(str(m) for m in sorted(df['maze'].unique()))

    lines = [
        "=== USV DDQN — TRAINING SUMMARY (multi-maze v_19_05) ===",
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Source    : {BASE_DIR.name}/training_log.csv",
        "",
        "--- CONFIG ---",
        f"MAX_STEPS    : {MAX_STEPS}",
        f"WINDOW       : {WINDOW}    (global rolling)",
        f"WINDOW_MAZE  : {WINDOW_MAZE}     (per-maze rolling, reactive a blocks)",
        f"MAZE(S)      : {maze_str}",
        "",
        "--- GLOBAL ---",
        f"Total episodes      : {total_ep}",
        f"Total steps         : {total_steps}",
        f"Epsilon (final)     : {eps_final}",
        f"Crash rate all      : {df['crashed'].mean() * 100:.1f}%",
        f"Crash rate last {WINDOW}  : {df['crashed'].tail(WINDOW).mean() * 100:.1f}%",
        f"Avg reward all      : {df['reward'].mean():.1f}",
        f"Avg reward last {WINDOW}  : {df['reward'].tail(WINDOW).mean():.1f}",
        "",
    ]

    for m in MAZES_TRAIN:
        sub = df[df['maze'] == m]
        if sub.empty:
            continue
        last_n = sub.tail(WINDOW_MAZE) if len(sub) >= WINDOW_MAZE else sub
        lines += [
            f"--- {MAZE_LABEL[m]} ---",
            f"Episodes              : {len(sub)}",
            f"Crash rate all        : {sub['crashed'].mean() * 100:.1f}%",
            f"Crash rate last {WINDOW_MAZE}  : {last_n['crashed'].mean() * 100:.1f}%",
            f"Avg reward all        : {sub['reward'].mean():.1f}",
            f"Avg reward last {WINDOW_MAZE}  : {last_n['reward'].mean():.1f}",
            f"Avg steps all         : {sub['steps'].mean():.1f}",
            f"Avg steps last {WINDOW_MAZE}   : {last_n['steps'].mean():.1f}",
            f"Max-steps survival rate (all) : {(sub['steps'] == MAX_STEPS).mean() * 100:.1f}%",
            "",
        ]

        if has_spawn:
            grp       = sub.groupby('spawn')
            avg_steps = grp['steps'].mean().sort_values(ascending=False)
            max_rate  = (sub['steps'] == MAX_STEPS).groupby(sub['spawn']).mean().reindex(avg_steps.index).fillna(0)
            uses      = grp.size()
            lines += [
                f"  Spawn breakdown {MAZE_LABEL[m]}:",
                f"  {'Spawn':<18} {'Uses':>5} {'Avg steps':>10} {'Max-steps rate':>15}",
                f"  {'-' * 52}",
            ]
            for spawn in avg_steps.index:
                lines.append(
                    f"  {spawn:<18} {uses[spawn]:>5} {avg_steps[spawn]:>10.1f} "
                    f"{max_rate.get(spawn, 0.0):>14.1%}"
                )
            lines.append("")

    lines.append("=== END ===")
    SUMMARY_TXT.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  Saved: summary_training.txt")


# ── Test plots ────────────────────────────────────────────────────────────────

def plot_test_maze(df_maze, maze_id):
    n_ep      = len(df_maze)
    has_spawn = 'spawn' in df_maze.columns
    color     = MAZE_COLORS[maze_id]

    global_success    = 1.0 - df_maze['crashed'].mean()
    global_avg_steps  = df_maze['steps'].mean()
    global_std_steps  = df_maze['steps'].std()

    if has_spawn:
        grp             = df_maze.groupby('spawn')
        spawn_success   = 1.0 - grp['crashed'].mean()
        spawn_avg_steps = grp['steps'].mean()
        spawn_std_steps = grp['steps'].std().fillna(0)
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
    colors = [color] + [plt.cm.Pastel1(i) for i in range(len(labels) - 1)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(x, succ_vals, color=colors, alpha=0.85)
    for bar, val in zip(bars1, succ_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f'{val:.0%}', ha='center', va='bottom', fontsize=9)
    ax1.set_ylabel('Success rate')
    ax1.set_title('Success rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, 1.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax1.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax1.grid(True, alpha=0.3, axis='y')

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

    fig.suptitle(f'Test — {MAZE_LABEL[maze_id]}  ({n_ep} episodes)',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    fname = f'{8 + maze_id:02d}_test_M{maze_id}.png'   # 09/10/11
    save_fig(fig, fname)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    setup_dirs()

    print("\n[Training]")
    df_train = load_training_csv()

    plot_reward_global(df_train)
    plot_reward_per_maze(df_train)
    plot_crash_per_maze(df_train)
    plot_steps_per_maze(df_train)

    for m in MAZES_TRAIN:
        plot_spawn_per_maze(df_train, m)

    plot_loss_curve(df_train)
    plot_maze_timeline(df_train)
    write_summary(df_train)

    print("\n[Test]")
    df_test = load_test_csv()
    if df_test is not None:
        for maze_id in MAZES_TESTED:
            df_maze = df_test[df_test['maze_id'] == maze_id]
            if len(df_maze) == 0:
                print(f"  No rows for M{maze_id} — skip.")
                continue
            plot_test_maze(df_maze, maze_id)

    print(f"\nDone. Output in: {BASE_DIR}\n")


if __name__ == '__main__':
    main()

"""
analisi.py — Analisi completa training_log.csv + test_results.csv
Branch: corrections_claude
Run:    python risultati/analisi.py
Output: risultati/plots/ (PNG) + stampa statistica su stdout
"""

import csv
import math
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
SCRIPTS_DIR  = os.path.join(REPO_ROOT, 'src', 'my_usv', 'scripts')
PLOTS_DIR    = os.path.join(SCRIPT_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(SCRIPTS_DIR, 'training_log.csv')
TEST_CSV  = os.path.join(SCRIPTS_DIR, 'test_results.csv')

MAZE_COLORS = {1: '#2196F3', 2: '#F44336', 3: '#4CAF50'}
MAZE_NAMES  = {1: 'Maze 1 (9a)', 2: 'Maze 2 (9b)', 3: 'Maze 3 (10) [test only]'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def moving_avg(values, window):
    result = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        result.append(float(np.mean(values[lo:i+1])))
    return result


def load_training(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'ep':       int(r['ep_global']),
                'maze':     int(r['maze']),
                'steps':    int(r['steps']),
                'reward':   float(r['reward']),
                'avg100':   float(r['avg100']),
                'epsilon':  float(r['epsilon']),
                'avg_loss': float(r['avg_loss']),
                'crashed':  int(r['crashed']),
                'total_steps': int(r['total_steps']),
            })
    return rows


def load_test(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append({
                'maze':      int(r['maze_id']),
                'episode':   int(r['episode']),
                'steps':     int(r['steps']),
                'reward':    float(r['reward']),
                'crashed':   int(r['crashed']),
                'min_lidar': float(r['min_lidar']),
                'avg_lidar': float(r['avg_lidar']),
            })
    return rows


# ---------------------------------------------------------------------------
# Training statistics
# ---------------------------------------------------------------------------
def print_training_stats(rows):
    print("\n" + "="*60)
    print("TRAINING LOG — STATISTICHE")
    print("="*60)

    total = len(rows)
    print(f"Episodi totali : {total}")
    print(f"Step totali    : {rows[-1]['total_steps']:,}")

    by_maze = defaultdict(list)
    for r in rows:
        by_maze[r['maze']].append(r)

    for maze in sorted(by_maze):
        eps     = by_maze[maze]
        n       = len(eps)
        crashes = sum(1 for e in eps if e['crashed'])
        success = n - crashes
        rewards = [e['reward'] for e in eps]
        ep_first = eps[0]['ep']
        ep_last  = eps[-1]['ep']
        print(f"\n  Maze {maze}  (ep {ep_first}→{ep_last})")
        print(f"    Episodi  : {n}")
        print(f"    Successi : {success} ({100*success/n:.1f}%)")
        print(f"    Crash    : {crashes} ({100*crashes/n:.1f}%)")
        print(f"    Reward   : min={min(rewards):.0f}  max={max(rewards):.0f}  "
              f"mean={np.mean(rewards):.0f}  median={np.median(rewards):.0f}")
        if success > 0:
            ok_rews = [e['reward'] for e in eps if not e['crashed']]
            print(f"    Reward successi: mean={np.mean(ok_rews):.0f}  std={np.std(ok_rews):.0f}")

    # Phase transition
    phase2_ep = None
    for r in rows:
        if r['maze'] == 2:
            phase2_ep = r['ep']
            break
    if phase2_ep:
        print(f"\n  Phase 2 iniziata a ep {phase2_ep}")

    # Epsilon at phase2
    eps_at_phase2 = next((r['epsilon'] for r in rows if r['ep'] == phase2_ep), None)
    if eps_at_phase2:
        print(f"  Epsilon al momento Phase 2: {eps_at_phase2:.4f}")

    # Final avg100
    print(f"\n  avg100 finale: {rows[-1]['avg100']:.1f}")
    print(f"  Epsilon finale: {rows[-1]['epsilon']:.4f}")


# ---------------------------------------------------------------------------
# Test statistics
# ---------------------------------------------------------------------------
def print_test_stats(rows):
    print("\n" + "="*60)
    print("TEST RESULTS — STATISTICHE (best_ddqn_model.pth, ε=0)")
    print("="*60)

    by_maze = defaultdict(list)
    for r in rows:
        by_maze[r['maze']].append(r)

    for maze in sorted(by_maze):
        eps     = by_maze[maze]
        n       = len(eps)
        success = sum(1 for e in eps if not e['crashed'])
        avg_steps  = np.mean([e['steps'] for e in eps])
        avg_reward = np.mean([e['reward'] for e in eps])
        avg_minl   = np.mean([e['min_lidar'] for e in eps])
        avg_avgl   = np.mean([e['avg_lidar'] for e in eps])
        print(f"\n  Maze {maze}  ({MAZE_NAMES[maze]})")
        print(f"    Episodi    : {n}")
        print(f"    Successi   : {success}/{n} ({100*success/n:.0f}%)")
        print(f"    Steps medi : {avg_steps:.1f}")
        print(f"    Reward med : {avg_reward:.1f}")
        print(f"    min_lidar  : {avg_minl:.3f} m")
        print(f"    avg_lidar  : {avg_avgl:.3f} m")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_reward_curves(rows):
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=False)
    fig.suptitle('Training Reward — corrections_claude branch', fontsize=13, fontweight='bold')

    eps_all    = [r['ep']     for r in rows]
    rew_all    = [r['reward'] for r in rows]
    avg100_all = [r['avg100'] for r in rows]
    maze_all   = [r['maze']   for r in rows]

    # ---- Plot 1: raw reward colored by maze ----
    ax = axes[0]
    by_maze = defaultdict(lambda: ([], []))
    for ep, rew, maze in zip(eps_all, rew_all, maze_all):
        by_maze[maze][0].append(ep)
        by_maze[maze][1].append(rew)
    for maze in sorted(by_maze):
        eps_m, rews_m = by_maze[maze]
        ax.scatter(eps_m, rews_m, s=2, alpha=0.4, color=MAZE_COLORS[maze], label=MAZE_NAMES[maze])
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Reward episodio')
    ax.set_title('Reward grezzo per episodio (colorato per maze)')
    ax.legend(markerscale=4, loc='upper left')
    ax.set_xlim(0, max(eps_all) + 10)

    # ---- Plot 2: avg100 ----
    ax = axes[1]
    ax.plot(eps_all, avg100_all, color='#333333', linewidth=1.2, label='avg100')
    ax.axhline(1500, color='orange', linewidth=1, linestyle='--', label='Phase2 threshold (1500)')
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    phase2_ep = next((r['ep'] for r in rows if r['maze'] == 2), None)
    if phase2_ep:
        ax.axvline(phase2_ep, color='red', linewidth=1, linestyle=':', label=f'Phase 2 start (ep {phase2_ep})')
    ax.set_ylabel('avg100')
    ax.set_title('Media mobile 100 episodi')
    ax.legend(loc='upper left')
    ax.set_xlim(0, max(eps_all) + 10)

    # ---- Plot 3: success rate per maze (rolling 100 ep) ----
    ax = axes[2]
    by_maze2 = defaultdict(list)
    for r in rows:
        by_maze2[r['maze']].append((r['ep'], 1 - r['crashed']))
    for maze in sorted(by_maze2):
        data = by_maze2[maze]
        eps_m  = [d[0] for d in data]
        succ_m = [d[1] for d in data]
        roll   = moving_avg(succ_m, min(100, len(succ_m)))
        ax.plot(eps_m, [r * 100 for r in roll], color=MAZE_COLORS[maze],
                linewidth=1.5, label=MAZE_NAMES[maze])
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Success rate % (rolling 100)')
    ax.set_title('Tasso di successo per maze (media mobile 100 ep)')
    ax.set_ylim(-5, 105)
    ax.legend(loc='upper left')
    ax.set_xlim(0, max(eps_all) + 10)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'training_reward.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"\n  [plot] Salvato: {out}")


def plot_epsilon_and_loss(rows):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('Epsilon decay e Loss — corrections_claude branch', fontsize=13, fontweight='bold')

    eps_all   = [r['ep']       for r in rows]
    eps_val   = [r['epsilon']  for r in rows]
    loss_val  = [r['avg_loss'] for r in rows]

    ax1.plot(eps_all, eps_val, color='#9C27B0', linewidth=1.2)
    ax1.set_ylabel('Epsilon')
    ax1.set_title('Epsilon decay (β=0.995)')
    phase2_ep = next((r['ep'] for r in rows if r['maze'] == 2), None)
    if phase2_ep:
        eps_at_p2 = next(r['epsilon'] for r in rows if r['ep'] == phase2_ep)
        ax1.axvline(phase2_ep, color='red', linewidth=1, linestyle=':',
                    label=f'Phase 2 start  ε={eps_at_p2:.3f}')
        ax1.legend()

    ax2.plot(eps_all, loss_val, color='#FF9800', linewidth=0.8, alpha=0.7)
    ax2.set_xlabel('Episodio')
    ax2.set_ylabel('avg_loss')
    ax2.set_title('Loss media per episodio')
    ax2.set_yscale('log')

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'epsilon_loss.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Salvato: {out}")


def plot_steps_per_episode(rows):
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle('Steps per episodio — corrections_claude branch', fontsize=13, fontweight='bold')

    by_maze = defaultdict(lambda: ([], []))
    for r in rows:
        by_maze[r['maze']][0].append(r['ep'])
        by_maze[r['maze']][1].append(r['steps'])

    for maze in sorted(by_maze):
        eps_m, steps_m = by_maze[maze]
        roll = moving_avg(steps_m, min(50, len(steps_m)))
        ax.scatter(eps_m, steps_m, s=2, alpha=0.25, color=MAZE_COLORS[maze])
        ax.plot(eps_m, roll, color=MAZE_COLORS[maze], linewidth=1.5, label=f'{MAZE_NAMES[maze]} (avg50)')

    ax.axhline(1000, color='gray', linewidth=0.8, linestyle='--', label='MAX_STEPS=1000')
    ax.set_xlabel('Episodio')
    ax.set_ylabel('Steps')
    ax.legend(loc='upper left', markerscale=4)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'steps_per_episode.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Salvato: {out}")


def plot_test_results(test_rows):
    by_maze = defaultdict(list)
    for r in test_rows:
        by_maze[r['maze']].append(r)

    mazes   = sorted(by_maze.keys())
    n_mazes = len(mazes)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Test Results — best_ddqn_model.pth  (ε=0)', fontsize=13, fontweight='bold')

    # Success rate
    ax = axes[0]
    success_rates = [100 * sum(1 for r in by_maze[m] if not r['crashed']) / len(by_maze[m]) for m in mazes]
    bars = ax.bar([MAZE_NAMES[m] for m in mazes], success_rates,
                  color=[MAZE_COLORS[m] for m in mazes], edgecolor='black')
    ax.set_ylabel('%')
    ax.set_title('Success rate (%)')
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, success_rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.0f}%', ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

    # Avg steps
    ax = axes[1]
    avg_steps = [np.mean([r['steps'] for r in by_maze[m]]) for m in mazes]
    bars = ax.bar([MAZE_NAMES[m] for m in mazes], avg_steps,
                  color=[MAZE_COLORS[m] for m in mazes], edgecolor='black')
    ax.set_ylabel('steps')
    ax.set_title('Steps medi per episodio')
    for bar, val in zip(bars, avg_steps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

    # avg_lidar
    ax = axes[2]
    avg_lidar = [np.mean([r['avg_lidar'] for r in by_maze[m]]) for m in mazes]
    bars = ax.bar([MAZE_NAMES[m] for m in mazes], avg_lidar,
                  color=[MAZE_COLORS[m] for m in mazes], edgecolor='black')
    ax.set_ylabel('m')
    ax.set_title('avg_lidar medio (distanza media da pareti)')
    for bar, val in zip(bars, avg_lidar):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}m', ha='center', va='bottom', fontweight='bold')
    ax.tick_params(axis='x', labelsize=8)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'test_results.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Salvato: {out}")


def plot_phase_zoom(rows):
    """Zoom sulla zona di transizione Phase 1→2 (ep 250-600)."""
    phase2_ep = next((r['ep'] for r in rows if r['maze'] == 2), None)
    if not phase2_ep:
        return

    lo, hi = max(1, phase2_ep - 250), min(rows[-1]['ep'], phase2_ep + 150)
    subset = [r for r in rows if lo <= r['ep'] <= hi]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.suptitle(f'Zoom transizione Phase 1→2 (ep {lo}–{hi})', fontsize=13, fontweight='bold')

    by_maze = defaultdict(lambda: ([], []))
    for r in subset:
        by_maze[r['maze']][0].append(r['ep'])
        by_maze[r['maze']][1].append(r['reward'])

    for maze in sorted(by_maze):
        eps_m, rews_m = by_maze[maze]
        ax.scatter(eps_m, rews_m, s=6, alpha=0.6, color=MAZE_COLORS[maze], label=MAZE_NAMES[maze])

    eps_sub = [r['ep'] for r in subset]
    avg_sub = [r['avg100'] for r in subset]
    ax.plot(eps_sub, avg_sub, color='black', linewidth=2, label='avg100', zorder=5)
    ax.axvline(phase2_ep, color='red', linewidth=1.5, linestyle='--',
               label=f'Phase 2 start (ep {phase2_ep})')
    ax.axhline(1500, color='orange', linewidth=1, linestyle=':', label='Phase2 threshold')
    ax.axhline(0, color='gray', linewidth=0.5)

    ax.set_xlabel('Episodio')
    ax.set_ylabel('Reward')
    ax.legend(loc='upper left', markerscale=3)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'phase_transition_zoom.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Salvato: {out}")


def plot_maze2_detail(rows):
    """Analisi dettagliata di maze 2: reward distribution e steps distribution."""
    maze2 = [r for r in rows if r['maze'] == 2]
    if not maze2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Maze 2 — Dettaglio training (Phase 2)', fontsize=13, fontweight='bold')

    # Reward distribution
    ax = axes[0]
    rewards = [r['reward'] for r in maze2]
    ax.hist(rewards, bins=40, color=MAZE_COLORS[2], edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rewards), color='black', linewidth=2, linestyle='--',
               label=f'mean={np.mean(rewards):.0f}')
    ax.axvline(0, color='gray', linewidth=1)
    ax.set_xlabel('Reward episodio')
    ax.set_ylabel('Frequenza')
    ax.set_title('Distribuzione reward (maze 2)')
    ax.legend()

    # Steps distribution
    ax = axes[1]
    steps = [r['steps'] for r in maze2]
    ax.hist(steps, bins=40, color=MAZE_COLORS[2], edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(steps), color='black', linewidth=2, linestyle='--',
               label=f'mean={np.mean(steps):.0f}')
    ax.axvline(1000, color='green', linewidth=1.5, linestyle=':', label='MAX_STEPS')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Frequenza')
    ax.set_title('Distribuzione steps per episodio (maze 2)')
    ax.legend()

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'maze2_detail.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Salvato: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Analisi risultati — corrections_claude branch")
    print(f"Training CSV : {TRAIN_CSV}")
    print(f"Test CSV     : {TEST_CSV}")
    print(f"Output plots : {PLOTS_DIR}")

    # Load
    if not os.path.exists(TRAIN_CSV):
        print(f"ERRORE: {TRAIN_CSV} non trovato"); sys.exit(1)
    if not os.path.exists(TEST_CSV):
        print(f"ERRORE: {TEST_CSV} non trovato"); sys.exit(1)

    train_rows = load_training(TRAIN_CSV)
    test_rows  = load_test(TEST_CSV)

    # Stats
    print_training_stats(train_rows)
    print_test_stats(test_rows)

    # Plots
    print("\n--- Generazione plots ---")
    plot_reward_curves(train_rows)
    plot_epsilon_and_loss(train_rows)
    plot_steps_per_episode(train_rows)
    plot_phase_zoom(train_rows)
    plot_maze2_detail(train_rows)
    plot_test_results(test_rows)

    print(f"\nDone. Tutti i plot in: {PLOTS_DIR}")


if __name__ == '__main__':
    main()

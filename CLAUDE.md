# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

A DDQN (Double Deep Q-Network) reinforcement learning agent that trains a simulated UGV (Unmanned Ground Vehicle) to navigate through maze-like environments while avoiding collisions. The simulation runs in Gazebo via ROS 2, entirely inside Docker.

## Environment Requirements

- Windows 11 + WSL2 + Ubuntu
- Docker Desktop (WSL2 backend) — must be running before any command
- VcXsrv/XLaunch for GUI (required for Gazebo; must enable "Disable access control")
- All commands below are run in **Git Bash** from the project root

## Build

```bash
# First time only: build Docker image (~2.5 GB, downloads ROS 2 Humble + PyTorch CPU)
docker build -t usv_rl_project .

# First time only: compile the ROS 2 package inside the container
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash
# then inside: colcon build && exit
```

After `colcon build`, the `build/` and `install/` directories are created on the host via the volume bind. Python/URDF changes propagate instantly — only structural ROS package changes (CMakeLists, package.xml) need a rebuild.

## Training

```bash
# Multi-maze training (M1+M2 ratio 1:2, 5000 episodes), seeded + per-config artifacts
./start_train_multimaze.sh --seed=0 --config=r_alpha

# Reset (BACKS UP runs/<config>/seed_<S>/ to ANALISI_TRAINING/<date>/ first, then wipes)
./start_train_multimaze.sh --seed=0 --config=r_alpha --reset
```

The script manages the Gazebo lifecycle automatically per 200-episode block. Artifacts go to `runs/<config>/seed_<S>/` (`checkpoint.pkl`, `best_model.pth`, `training_log.csv`); block logs to `logs/multimaze_block_*.log`. **Always pass `--seed` and `--config`** — see `guida_operativa_seria.md` for the multi-seed protocol and the 4-PC split.

## Testing

```bash
# Round-robin eval on all 3 mazes, reproducible, rich per-step logging
./start_test.sh --seed=0 --config=r_alpha --reps=30
```

Evaluates `runs/<config>/seed_<S>/best_model.pth` on all 3 mazes. Spawns are covered **round-robin** (each spawn exactly `--reps` times → M1=2×reps, M2=6×reps, M3=1×reps), so coverage is balanced and reproducible. ε=0.0 (greedy). Outputs in `runs/<config>/seed_<S>/`:

| File | Content |
|---|---|
| `eval_summary.csv` | one row per maze: success_rate, avg_reward, avg_steps (consumed by `aggregate_seeds.py`) |
| `eval_steps_m<N>.csv` | per-step: action, q_chosen/q_max/q_spread, front/left/right/min_lidar, reward, done (add `--log-q-full` for all 11 Q-values) |
| `eval_crashes_m<N>.csv` | per crash: crash_sector, crash_dist, last 5 actions (diagnostica) |
| `run_meta.json` | provenance: seed, git_sha, hostname, timestamp, success criterion |

**Aggregate multiple seeds** (report mean±std / IQM / 95% bootstrap CI — never the max):

```bash
python3 src/my_usv/scripts/aggregate_seeds.py \
  --config r_alpha --output ANALISI_TRAINING/$(date +%Y_%m_%d)/aggregate_r_alpha.csv
```

## Manual Container

```bash
# Enter the container interactively
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash

# Run train.py directly (inside container, after sourcing setup)
source install/setup.bash
python3 src/my_usv/scripts/train.py --start-ep 0 --end-ep 100 --maze-id 1 --checkpoint src/my_usv/scripts/checkpoint.pkl

# Kill stuck container
docker rm -f usv_container
```

## Architecture

### Core files (`src/my_usv/scripts/`)

| File | Role |
|---|---|
| `usv_env.py` | ROS 2 node — the RL environment. Handles Gazebo reset, LIDAR processing, reward computation, and action publishing |
| `ddqn_model.py` | Neural network: 50 → 300 → 300 → 11 (ReLU, fully connected) |
| `train.py` | DDQN training loop with replay buffer and target network. CLI `--seed`; logs episode rows + `crash_sector` |
| `test.py` | Greedy eval (ε=0.0). Round-robin spawns, per-step + crash + summary CSV logging. CLI `--seed --reps --config --out-dir --log-q-full` |
| `seeding.py` | `set_global_seed(seed)` — fixes random/numpy/torch RNG. Call FIRST in any entrypoint |
| `aggregate_seeds.py` | Multi-seed reducer → mean±std / IQM / 95% bootstrap CI per maze |
| `usv_logic.py` | Pure reward/lidar logic + shared sector helpers (`sector_distances`, `crash_sector`, `round_robin_spawn`) |
| `patch_world.py` | Rewrites `<real_time_update_rate>` in `.world` files to control sim speed |

### State / Action / Reward

- **State:** 50 normalized LIDAR bins (raw 512 rays → min-pooled → divided by 5.0m), range [0, 1]
- **Actions:** 11 discrete angular velocities from -0.8 to +0.8 rad/s; linear velocity fixed at 0.5 m/s
- **Reward:** +5 per step, −0.1×|action−center| steering penalty, −20×severity³ front danger (<1.5m), −5×severity² side danger (<0.45m), −1000 collision (<0.25m, ends episode)
- **Episode limit:** 500 steps

### Training hyperparameters

| Parameter | Value |
|---|---|
| Discount γ | 0.99 |
| Learning rate | 0.00025 (Adam) |
| Batch size | 64 |
| Replay buffer | 100,000 transitions |
| Target net update | every 1,000 global steps |
| ε decay | ×0.995 per episode (1.0 → ~0.05) |

### Maze split

- **Maze 1** (`labirinto_9a.world`, spawn x=-3, y=-5, yaw=1.57) — training
- **Maze 2** (`labirinto_9b.world`, spawn x=-6, y=0, yaw=0) — training
- **Maze 3** (`labirinto_10.world`, spawn x=-2, y=-1, yaw=0) — test only, never seen during training

### Checkpoint format

`checkpoint.pkl` (pickle) contains: Q-net weights, target-net weights, optimizer state, full replay buffer, epsilon, global step count, reward history, crash count, **seed**. Written atomically via `.tmp` rename. ~40 MB.

### Reproducibility & evaluation rigor

Seeds are controlled (`seeding.set_global_seed`, `--seed` on train/test) so variance is **attributable** — but Gazebo physics/timing stays non-deterministic, so there is no bit-for-bit reproducibility. Conclusions need **≥3–5 seeds** of the same config; report distributions (mean±std / IQM / CI), never single runs or the max (Henderson 2018, Agarwal 2021). Artifacts are isolated per `runs/<config>/seed_<S>/`. `runs/` is git-ignored — version only the aggregated CSVs under `ANALISI_TRAINING/`. **The full operating protocol (seeds, 4-PC split, reporting) is in `guida_operativa_seria.md`.**

### ROS 2 topics

- `/scan` (LaserScan, subscribed) — LIDAR input
- `/cmd_vel` (Twist, published) — velocity commands
- `/reset_world` (Empty service) — resets Gazebo world

### Key timing constraint

Simulation runs at `GAZEBO_SPEED=4` (4× real-time). Each action step waits 0.1s sim-time via `_wait_sim_seconds()`. After reset, the env drains 20 spin cycles to flush stale LIDAR messages before accepting new observations. Reducing speed below 3× causes LIDAR sync failures.

## Development Workflow

1. Edit Python files on Windows — changes are live inside the container immediately.
2. To test a change: `Ctrl+C` in the training terminal, re-run `./start_training_curriculum.sh`.
3. Gazebo does not need to be restarted for Python-only changes.
4. `colcon build` is only needed when `CMakeLists.txt`, `package.xml`, or URDF files change.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

A DDQN (Double Deep Q-Network) reinforcement learning agent that trains a simulated USV (Unmanned Surface Vehicle) to navigate through maze-like environments while avoiding collisions. The simulation runs in Gazebo via ROS 2, entirely inside Docker.

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
# Start or resume curriculum training (3000 episodes, Maze 1 & 2 alternating)
./start_training_curriculum.sh

# Reset all checkpoints and restart from scratch
./start_training_curriculum.sh --reset
```

The script manages Gazebo lifecycle automatically per 100-episode block. Logs go to `logs/block_N_maze_M.log`. Training state is in `src/my_usv/scripts/curriculum_state.txt`.

## Testing

```bash
./start_test.sh
```

Evaluates the best saved model on all 3 mazes (30 episodes each). Results in `src/my_usv/scripts/test_results.csv`.

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
| `train.py` | DDQN training loop with replay buffer and target network |
| `test.py` | Greedy policy evaluation (ε=0.0) |
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

`checkpoint.pkl` (pickle) contains: Q-net weights, target-net weights, optimizer state, full replay buffer, epsilon, global step count, reward history, crash count. Written atomically via `.tmp` rename. ~40 MB.

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

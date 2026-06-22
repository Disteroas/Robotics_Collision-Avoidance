# Robotics — Collision Avoidance for a Simulated USV (DDQN)

A Double Deep Q-Network (DDQN) reinforcement-learning agent that trains a simulated
Unmanned Surface Vehicle (USV) to navigate maze-like environments while avoiding
collisions. The simulation runs in **Gazebo** via **ROS 2 Humble**, fully containerised
with **Docker**. The agent perceives the world through a 2D LIDAR and outputs discrete
steering actions.

> **Where is the final work?** This `main` branch is the **original baseline**
> (single-maze training, basic reward). The complete, rigorous study — faithful paper
> reproduction, multi-seed evaluation, and the final report — lives on the
> `paper_metric_feng` and `paper_metric_base` branches. See **[BRANCHES.md](BRANCHES.md)**
> for a map of every branch and what it contains.

---

## What the project does

- **Task:** drive a USV through three maze worlds without hitting the walls.
- **Method:** DDQN with experience replay and a target network.
- **State:** normalised LIDAR scan (range-limited and down-pooled into bins).
- **Actions:** a discrete set of angular velocities; linear speed is held constant.
- **Reward:** survival reward with danger/steering penalties and a large collision penalty.
- **Three mazes:** two used for training, one held out for testing generalisation.

The full architecture, hyper-parameters, state/action/reward definitions, and the
multi-seed evaluation protocol are documented in **`CLAUDE.md`** and
**`guida_operativa_seria.md`** on the study branches.

---

## Requirements

- Windows 11 + WSL2 + Ubuntu (or any Linux with Docker)
- Docker Desktop (WSL2 backend) — must be running before any command
- An X server (e.g. VcXsrv/XLaunch on Windows) for the Gazebo GUI
- Commands below are run from the project root

A complete, step-by-step environment setup (BIOS virtualisation, WSL2, X11, Docker)
is in **[START_HERE.md](START_HERE.md)**. Day-to-day operation is in
**[GUIDA_OPERATIVA.md](GUIDA_OPERATIVA.md)**.

---

## Quick start (this baseline branch)

```bash
# 1) Build the Docker image (first time only: downloads ROS 2 + PyTorch CPU)
docker build -t usv_rl_project .

# 2) Compile the ROS 2 package (first time only)
docker run -it --rm --volume="/$(pwd):/home/usv_ws" usv_rl_project bash
#   inside the container:
colcon build && exit

# 3) Launch the simulator / train / test
./start_sim.sh            # Gazebo with GUI
./start_sim_headless.sh   # headless (no GUI)
./start_train.sh          # train the agent
./start_test.sh           # evaluate a trained model
```

Python and URDF changes propagate into the container instantly through the volume
mount; only structural ROS package changes (`CMakeLists.txt`, `package.xml`) require a
rebuild.

For the **final study branches**, the entry points are richer (seeded multi-maze
training, round-robin evaluation, multi-seed aggregation). See those branches' own
guides and `CLAUDE.md`.

---

## Repository layout (baseline branch)

| Path | Content |
|---|---|
| `src/my_usv/` | ROS 2 package: RL environment node, DDQN model, train/test scripts |
| `Dockerfile` | ROS 2 Humble + PyTorch CPU image definition |
| `start_*.sh` | Helper scripts to launch simulation, training and testing |
| `labirinti_coordinate.md` | Spawn coordinates and world files for the three mazes |
| `script_Davide_python/` | Standalone DDQN tutorial reference |
| `START_HERE.md`, `GUIDA_OPERATIVA.md` | Setup and operating guides |

---

## Reproducibility

Because Gazebo physics and timing are not bit-for-bit deterministic, single runs are
**not** reliable evidence. The study branches fix the RNG seeds (`--seed`) so that
variance is attributable, and report **distributions over multiple seeds**
(mean ± std / IQM / bootstrap CI) rather than a single best run. The full protocol is
documented on the `paper_metric_feng` branch.

---

## Authors

Bolognini · Covolo · D'Antona · Masciavè

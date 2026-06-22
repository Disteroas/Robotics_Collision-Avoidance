# Branch guide

This repository keeps the full development history as separate branches: each one is a
snapshot of a specific experiment or milestone. This file maps what every branch
contains so the work can be navigated without guessing from commit logs.

## Where to look first

| If you want… | Go to |
|---|---|
| The **final implementation, multi-seed study and report** (faithful paper reproduction) | **`paper_metric_feng`** |
| The **enhanced reward variant** used as comparison in the final report | **`paper_metric_base`** |
| The **original baseline** and the environment setup guides | **`main`** |

## Main branches

| Branch | Date | What it is |
|---|---|---|
| `main` | 2026-05 | **Original baseline.** Single-maze DDQN, ROS 2 / Gazebo / Docker setup, environment and operating guides. Starting point of the project. |
| `paper_metric_feng` | 2026-06 | **Final — faithful reproduction.** Feng-style configuration (compact LIDAR state), multi-seed training campaign and round-robin evaluation, aggregated results, figures and the final report *"Beyond the Best Run"*. Primary deliverable. |
| `paper_metric_base` | 2026-06 | **Final — enhanced reward variant** (`r_alpha`, shaped reward). Used as the comparison agent in the feng-vs-`r_alpha` study. Companion to `paper_metric_feng`. |

## Development milestones

| Branch | Date | What it is |
|---|---|---|
| `paper_implementation` | 2026-05 | Rewrite of the training/environment code toward the reference paper. |
| `feng_direct` | 2026-05 | Direct Feng-parameter multi-maze interleaved training experiments. |
| `fixed_feng` | 2026-05 | "Fixed Feng" variant; its failure analysis informed later design choices. |
| `ddqn_en_18_05` | 2026-05 | DDQN-enhanced, design Round 1 analysis. |
| `ddqn_en_19_05` | 2026-05 | DDQN-enhanced, Round 2 — reward recalibration (`r_alpha`); analysis folders reorganised. |
| `ddqn_en_20_05` | 2026-05 | DDQN-enhanced, training-time / test-count fixes. |
| `curriculum_learning` | 2026-05 | Curriculum-learning approach with training/test analysis tooling. |
| `gym_env` | 2026-05 | Port toward a Gym-style environment interface (+ usage guide). |
| `fast_sim` | 2026-04 | Faster / accelerated simulation experiments and speed notes. |

## Iterative training snapshots ("merge" series)

Dated checkpoints of the multi-maze training pipeline, each isolating one change.

| Branch | Date | What it is |
|---|---|---|
| `merge11_05` | 2026-05 | Aligned to a reduced, validated M1 spawn set. |
| `merge12_05` | 2026-05 | Spec for replay-start-size + M2-only training + spawn logging. |
| `merge14_05` | 2026-05 | 8000-episode M2-only training spec, fresh start, 7 spawns. |
| `merge15_05` | 2026-05 | Per-spawn breakdown in terminal and report, prominent success rate. |
| `merge16_05` | 2026-05 | Validated Maze 3 spawn relocation. |

## Teammate branches

| Branch | Date | What it is |
|---|---|---|
| `matte_merge17_05` | 2026-05 | Yaw-handling experiments and analysis; PPO (stable-baselines3) track briefing. |
| `matte_merge21_05` | 2026-05 | Adjusted spawns and action-clipping variations. |
| `ila18_05` | 2026-05 | Training/testing snapshot with yaw in the state. |

# merge15_05 Training Design

## Goal

Train USV DDQN on Maze 2 only, 8000 episodes, with 7 training spawn points (3 removed from merge14_05 based on 3-run empirical analysis). Clean start from scratch. Target: M2≥50%, M3≥40% zero-shot generalization.

## Context — findings from merge14_05 (3 runs)

Three independent runs with identical config confirmed:

1. **Non-convergence at 4000 ep.** Peak avg100 occurs at ep 3115/3587/3818 across runs — the curve is still rising at ep 4000 in run3. Training must be extended.

2. **Spawn difficulty is structurally determined, not stochastic.** Classification is identical across 3 runs on 2 machines:
   - Class A (easy): F1 (-4.5,-3.5) 38-45%, F3 (6.0,6.0) 23-30% completion
   - Class C (long-fail): F2 (-1.5,-4.0) ~0% completion but avg 395 steps — valuable signal
   - Class E (catastrophic): D3 (3.5,0.5) avg 55 steps, 0/~1180 episodes completed — pure noise
   - Class D-short: D2 (0.5,-2.0) avg 110 steps, E2 (0.0,3.5) avg 128 steps — 0% across all 3 runs

3. **M3 generalization is achievable.** Run3 (final avg100=700, still rising) achieved M3=13% zero-shot with no M3 training. Hypothesis: generalization emerges above a training quality threshold (~avg100≥700 stable). More training → higher threshold reached → better M3 generalization.

4. **High test variance with 30 ep.** M2 test: 0%/30%/20% with identical config. 30 episodes (5 per spawn) gives CI ±18pp. Increase to 90 ep per maze in next test session.

## Configuration changes from merge14_05

| Parameter | merge14_05 | merge15_05 | Rationale |
|---|---|---|---|
| Total episodes | 4000 | **8000** | Curve not converged at ep 4000 |
| TOTAL_BLOCKS | 20 | **40** | 40 × 200 ep = 8000 |
| M2 training spawns | 10 | **7** | Remove 3 Class D/E spawns |
| Starting point | from scratch | **from scratch** | Scientific validity — clean config |

## Unchanged parameters

| Parameter | Value |
|---|---|
| BLOCK_SIZE | 200 ep |
| BLOCK_PATTERN | (2) — M2 only |
| REPLAY_START_SIZE | 10,000 |
| BETA_DECAY | 0.999 (ε: 1.0→0.05 over ~3000 ep) |
| MAX_STEPS | 500 |
| Reward function | +5/step, -1000 collision |
| Learning rate | 0.00025 (Adam) |
| GAZEBO_SPEED | 5× |

## Spawn changes — SPAWN_LISTS[2] in usv_env.py

### Remove (3 spawns)

| Spawn | Label | Avg steps (3-run) | Max-steps rate | Reason |
|---|---|---|---|---|
| (3.5, 0.5, 4.712) | D3 | **55** | 0% | Near-instant crash every episode. Probable wall-facing start. 1180+ episodes = zero completed. Pure gradient noise. |
| (0.5, -2.0, 1.571) | D2 | **110** | ~0% | Short crash, 0% completion across all 3 runs. Also in TEST_SPAWN_LISTS — stays there. |
| (0.0, 3.5, 3.142) | E2 | **128** | 0% | Short crash, 0% completion across all 3 runs. Also in TEST_SPAWN_LISTS — stays there. |

### Keep (7 spawns)

| Spawn | Label | Avg steps | Max-steps rate | Rationale |
|---|---|---|---|---|
| (-6.0, 0.0, 0.0) | A1 | 186-220 | 1.5-8.4% | Improving across runs (1.5%→6.9%→8.4%). Keep — in sblocco. |
| (-4.5, 1.5, 2.356) | B3 | 220-229 | 0% | Medium-length episodes, maze diversity. Keep. |
| (-7.0, 5.0, 0.0) | C2 | 297-300 | 0% | Medium-long episodes, consistent avg steps. Keep. |
| (1.5, 0.0, 3.142) | D1 | 77-125 | 0-5.1% | Borderline but some completion signal. Keep. |
| (-4.5, -3.5, 0.0) | F1 | 335-365 | **38-45%** | Primary learning spawn. Critical — keep. |
| (-1.5, -4.0, 1.571) | F2 | 375-402 | ~0% | Highest avg steps of all spawns. Robot traverses 75-80% of maze before bottleneck. Richest training signal per episode. Keep. |
| (6.0, 6.0, 3.142) | F3 | 325-348 | **23-30%** | Secondary learning spawn. Keep. |

**TEST_SPAWN_LISTS[2] unchanged** — test spawns are a separate list, not affected by training spawn removal.

## Files to modify

| File | Change |
|---|---|
| `start_train_multimaze.sh` | `TOTAL_BLOCKS=40` (was 20) |
| `src/my_usv/scripts/usv_env.py` | Remove D2, D3, E2 from `SPAWN_LISTS[2]` |
| `DOCUMENTAZIONE/ESPERIMENTI.md` | Add Esperimento 9 entry |

## Epsilon behavior at 8000 ep

With BETA_DECAY=0.999, ε reaches 0.05 at ep ~2995 (unchanged from merge14_05).
- ep 1-3000: ε 1.0→0.05 (exploration/exploitation transition)
- ep 3001-8000: ε=0.05 (5000 ep of exploitation vs 1000 ep in merge14_05)

More exploitation time benefits the policy once it learns — no change needed to BETA_DECAY. Keeping it unchanged preserves comparability with previous experiments.

## Prefill behavior

REPLAY_START_SIZE=10k with 7 spawns (vs 10 previously):
- Prefill episodes ≈ 10000/218 ≈ 46 episodes (similar to merge14_05 ~155 ep, but avg steps per ep may differ slightly without the very-short spawn crashes pulling down the average)
- Buffer diversity: 7 spawns × ~6-7 prefill ep per spawn — adequate for initial diversity

## Success criteria

| Metric | Target | merge14_05 best (run3) |
|---|---|---|
| M2 test success | ≥ 50% | 30% |
| M3 test success (zero-shot) | ≥ 40% | 13% |
| Final avg100 | > 900 stable | 700 (still rising) |
| Crash rate last 100 ep | < 80% | 81% |

## Test protocol

Run test with **90 episodes per maze** (was 30) to reduce CI from ±18pp to ±10pp. Use patched `test.py` (spawn logging active since commit 445fbd6) for per-spawn breakdown in test results.

## Branch

`merge15_05` (from `merge14_05`)

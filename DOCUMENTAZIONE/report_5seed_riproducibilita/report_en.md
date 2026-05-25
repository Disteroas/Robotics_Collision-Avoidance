# A Five-Seed Reproducibility Study of a DDQN Collision-Avoidance Agent

*Working paper — Machine A, single host. Italian informal companion: [`report_it.md`](report_it.md).*
*Configuration `r_alpha` · seeds 0–4 · 5000 training episodes each · greedy evaluation (ε = 0) · 2026-05-25.*

---

## Abstract

We evaluate a Double Deep Q-Network (DDQN) agent for unmanned-surface-vehicle (USV) maze navigation across five training runs that are identical in code and hyperparameters and differ **only in the random seed**. Despite this controlled setup, evaluation performance varies substantially across seeds. On the held-out maze (Maze 3) the outcome is **bimodal**: two seeds reach a 100% success rate while one collapses to 0%, yielding a mean of 67.3% with a 95% bootstrap confidence interval 66 percentage points wide. We show that (i) a single run is uninformative and the maximum is not a valid statistic (Henderson et al. 2018; Agarwal et al. 2021), (ii) training reward does not predict generalization — the seed with the second-highest training reward collapses at evaluation — and (iii) with n = 5 the confidence intervals remain too wide for tight quantitative claims. Hardware is anonymized; a cross-hardware comparison is reported only as a preliminary, single-run-per-machine observation.

---

## 1. Introduction

Deep reinforcement learning results are notoriously sensitive to random seeds, and the field has converged on reporting **distributions over multiple seeds** rather than single runs or best-of-N (Henderson et al. 2018; Islam et al. 2017; Agarwal et al. 2021). This report applies that protocol to our DDQN USV agent. The seed is treated as a **nuisance variable to be marginalized**, not a hyperparameter to be tuned.

---

## 2. Experimental Setup

| Item | Value |
|---|---|
| Algorithm | DDQN, MLP 50→300→300→11 (ReLU) |
| Configuration | `r_alpha` |
| Seeds | 0, 1, 2, 3, 4 (fixed via `set_global_seed`) |
| Training | 5000 episodes/seed (interleaved M1+M2 blocks, ratio 1:2) |
| Evaluation | round-robin over spawns, 30 repetitions, **ε = 0** (greedy) |
| Eval episodes | Maze 1 = 60 (2 spawns × 30), Maze 2 = 180 (6 × 30), Maze 3 = 30 (1 × 30) |
| Success criterion | reach **500 steps without collision** |
| Hardware | Machine A (single host) |

**Provenance / confound check.** From `run_meta.json`: seed 0 was trained at git SHA `3533c8a`, the remaining four at `474a363`. The diff between these commits is **documentation only** (no change to training or evaluation code), so there is **no confound**.

**Maze split.** Mazes 1 and 2 are seen during training; **Maze 3 is held out** (never seen during training). Maze 1 has 2 spawns, Maze 2 has 6, Maze 3 has a single spawn.

---

## 3. Results

Following Henderson et al. (2018) and Agarwal et al. (2021), we report **the full distribution — never the maximum**: mean ± standard deviation, the **Inter-Quartile Mean (IQM)** (robust to outliers), and a **95% bootstrap confidence interval**.

| Maze | Per-seed (s0/s1/s2/s3/s4) | Mean ± std | IQM | 95% CI | CI width |
|---|---|---|---|---|---|
| **M1** | 50 / 23.3 / 0 / 50 / 50 | **34.7% ± 22.6** | 41.1% | [14.7, 50.0] | 35 pt |
| **M2** | 66.7 / 37.8 / 50 / 33.3 / 32.2 | **44.0% ± 14.5** | 40.4% | [33.8, 56.7] | 23 pt |
| **M3** | 83.3 / 100 / 0 / 53.3 / 100 | **67.3% ± 42.2** | 78.9% | [30.7, 96.7] | **66 pt** |

![Per-seed success rate](figures/fig1_success_per_seed.png)

*Figure 1 — Per-seed success rate (coloured markers) with mean and 95% CI (black bars). Each marker is one seed. On Maze 3 the markers span the full 0–100% range, with two seeds saturating at 100% and one collapsing to 0% — a heavy-tailed, near-bimodal spread rather than noise around a single mean.*

**Key observations.**
- **Large across-seed variance**, greatest on Maze 3 (σ = 42 pt). A single run could have reported "M3 = 100%" (seeds 1 or 4) **or** "M3 = 0%" (seed 2) — opposite conclusions from identical code.
- **IQM > mean** on every maze: the mean is pulled down by the low tail (chiefly seed 2); the robust central tendency is higher.
- **Wide CIs**, especially Maze 3 (66 pt): with n = 5, no tight point estimate of Maze-3 success can be claimed.

### 3.1 Per-seed detail (reward and steps)

| Seed | M1 succ / reward / steps | M2 succ / reward / steps | M3 succ / reward / steps |
|---|---|---|---|
| 0 | 50% / 580.7 / 342.5 | 66.7% / 2054.0 / 455.4 | 83.3% / 2050.9 / 455.5 |
| 1 | 23.3% / 104.5 / 285.2 | 37.8% / 1544.9 / 426.0 | **100%** / 2665.1 / 500.0 |
| 2 | **0%** / −379.4 / 160.4 | 50% / 1535.2 / 422.5 | **0%** / 145.3 / 247.9 |
| 3 | 50% / 803.6 / 298.3 | 33.3% / 1316.9 / 408.1 | 53.3% / 1343.9 / 383.4 |
| 4 | 50% / 667.1 / 345.9 | 32.2% / 1236.7 / 382.5 | **100%** / 2022.0 / 500.0 |

![Reward and step distributions](figures/fig4_reward_steps_box.png)

*Figure 4 — Across-seed distribution of mean reward and survived steps per maze. Seed 2 (red) is the low outlier on M1 and M3; on M3 the split between seeds that saturate the 500-step limit and those that crash early is clearly visible.*

---

## 4. Failure-Mode Analysis

![Crash sectors](figures/fig3_crash_sectors.png)

*Figure 3 — Collision sector (summed over the 5 seeds) per maze. Maze 2 has more absolute crashes because it has more evaluation episodes (180 vs 60 vs 30).*

Crashes on Mazes 1 and 2 are **front-dominated**, consistent with a kinematic failure mode: the agent approaches an obstacle too directly to turn away in time, given the fixed 0.5 m/s linear velocity and limited turning radius. Maze 3 produces few crashes, concentrated in the **right** sector — a more localized failure tied to its single spawn geometry.

---

## 5. Discussion

### 5.1 Training reward does not predict generalization

![Training curves](figures/fig2_training_curves.png)

*Figure 2 — avg100 (mean reward over the last 100 episodes) vs episode for the 5 seeds. The sawtooth reflects the alternation of M1/M2 blocks. Seed 2 (thick red) is among the highest in training reward yet collapses at evaluation.*

| Seed | Training reward (mean of last 500 ep) | M3 eval |
|---|---|---|
| 0 | 1221 | 83.3% |
| 2 | 1130 | **0%** |
| 4 | 1099 | 100% |
| 1 | 820 | 100% |
| 3 | 692 | 53.3% |

The training-reward ranking **does not predict** the evaluation ranking: seed 2 is second by training reward yet last (0%) at evaluation. **Implication:** training reward is not a reliable model-selection signal for this task; selection and reporting must rely on greedy (ε = 0) evaluation and, crucially, on **multiple seeds**.

### 5.2 Why a single run is misleading

The five seeds share code, hyperparameters, and machine; the only varied factor is the seed, so the observed dispersion is **entirely** seed variance (plus evaluation/Gazebo stochasticity). Reporting the best seed (M3 = 100%) would constitute p-hacking (Henderson et al. 2018); reporting an arbitrary single run would be equally misleading given the bimodality.

### 5.3 On seeds

The seed is **not a hyperparameter** but a nuisance variable to marginalize. No intrinsically and transferably "better" seeds exist; some perform better *post hoc* through the interaction of weight initialization (the lottery-ticket effect, Frankle & Carbin 2019) and the exploration trajectory (a feedback loop that makes RL more seed-sensitive than supervised learning). Full treatment in [`PAPER_ANALYSIS/riproducibilita_seed_hardware.md`](../PAPER_ANALYSIS/riproducibilita_seed_hardware.md).

### 5.4 Cross-hardware (PRELIMINARY — do not conclude)

> **Status:** preliminary observation, **n = 1 per machine**. To be completed with the Machine B campaign before any conclusion.

The same seed (1), same code, same git SHA produced opposite Maze-3 outcomes on two machines: **Machine A ≈ 100%** vs **Machine B ≈ 3.3%**. Hypothesized mechanism: Gazebo non-determinism (physics/timing coupled to wall-clock) → a spawn-retry loop consuming a different number of RNG draws → desynchronization of the global RNG stream → chaotic amplification via DDQN bootstrapping (replay buffer + target network). Seeds **cannot be pooled across heterogeneous machines as i.i.d.** Details and references: [`PAPER_ANALYSIS/riproducibilita_seed_hardware.md`](../PAPER_ANALYSIS/riproducibilita_seed_hardware.md).

---

## 6. Limitations

1. **n = 5 is underpowered.** CIs are wide (M3: 66 pt). CI width scales as 1/√n: halving it requires ~20 seeds; a tight Maze-3 CI (±10 pt) would require ~30–40 seeds (infeasible here). See Colas et al. (2018) on power analysis.
2. **Single machine.** Results hold for Machine A; the cross-hardware comparison is incomplete.
3. **Evaluation noise not isolated.** The reported variance mixes training variance and evaluation (Gazebo) stochasticity; per-seed evaluations were not repeated to separate them. On Maze 3 the per-seed outcomes concentrate toward the extremes (two seeds at 100%, one at 0%) but two are intermediate (83.3%, 53.3%), so within-seed evaluation noise is not negligible. The large across-seed spread (0–100%) is nonetheless dominated by training variance; a clean separation would require repeated per-seed evaluations.
4. **Outlier retained.** Seed 2 is kept in the sample; dropping it would inflate results (p-hacking).

---

## 7. Conclusions

Configuration `r_alpha` on Machine A achieves, over 5 seeds, **M1 34.7% ± 22.6**, **M2 44.0% ± 14.5**, **M3 67.3% ± 42.2** (IQM 41.1 / 40.4 / 78.9). The salient finding is **not the mean but the instability**: on the held-out Maze 3 the policy collapses in roughly 1 of 5 seeds (bimodal 0/100). Training reward does not predict generalization, so model selection and reporting must use multi-seed greedy evaluation. With n = 5, distributions must be reported in full (as done here); tight quantitative claims and any cross-hardware statement require, respectively, more seeds and completing the Machine B campaign.

---

### Reproducing the figures

```bash
python DOCUMENTAZIONE/report_5seed_riproducibilita/make_figures.py
# sources: runs/r_alpha/seed_{0..4}/  ·  aggregate: ANALISI_TRAINING/2026_05_25/aggregate_r_alpha.csv
```

### References
Henderson et al. 2018 (arXiv:1709.06560); Islam et al. 2017 (arXiv:1708.04133); Agarwal et al. 2021 (arXiv:2108.13264); Colas et al. 2018 (arXiv:1806.08295); Frankle & Carbin 2019 (arXiv:1803.03635). Extended bibliography in [`PAPER_ANALYSIS/riproducibilita_seed_hardware.md`](../PAPER_ANALYSIS/riproducibilita_seed_hardware.md).

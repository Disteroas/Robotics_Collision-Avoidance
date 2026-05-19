# DDQN Round 2 — Reward Recalibration (R-α)

**Branch target:** `ddqn_en_20_05` (creato da `ddqn_en_19_05`)
**Date:** 2026-05-19
**Author:** Davide Covolo
**Predecessor:** Round 1 (ddqn_en_19_05) — M2=82% / M1=0% / M3=0%

---

## 1. Context

Il training Round 1 (multi-maze M1+M2 ratio 1:2, DR LIDAR σ=0.02m, D1 relocato) ha prodotto **best-ever M2=82%** ma M1=0% in test. Analisi geometrica + analisi reward hanno identificato due cause:

1. **Front sector geometricamente errato.** `front_dist = min(scan[15:35])` copre ±54° dall'asse forward. In corridoi M1 da 1.50m, side walls a ±54° vengono campionati a 0.93m → trigger costante front_danger (4.2/step) in corridoio dritto, anche quando nessun ostacolo frontale. M1 reward per-step su corridoio sano scende a +1.3 invece di +5.5.
2. **Penalty quadratic con weights elevati.** `20·severity²` (front), `5·severity²` (side) creano cliff vicino a collision (severity³ effetto). Sebbene il segnale forte abbia aiutato M2, in M1 amplifica il problema #1: il phantom front penalty raggiunge 4.2 con weight=20.

Round 2 corregge entrambe in un singolo cambio coerente ("reward calibration").

## 2. Goal

Aumentare success rate M1 da 0% verso almeno 30% mantenendo M2 ≥ 75%. Miglioramento M3 considerato benefit secondario non garantito.

**Non-goals:**
- Cambiare shape della penalty (quadratic resta).
- Aggiungere goal/distance/smoothness reward.
- Modificare hyperparams DDQN (LR, γ, batch, target update).
- Toccare DR_NOISE_STD.
- Modificare spawn lists.

## 3. Hypothesis

**H1 (primary):** Restringere il front sector a [20:30] (±27°, 54° totale) elimina il phantom front_danger nei corridoi M1 da 1.5m. Step reward M1 normalizza da +1.3 a +5.5/step. Policy può finalmente apprendere traversata corridoio P1 e manovra inner chamber P2.

**H2 (secondary):** Ridurre weights front 20→10 e side 5→3 bounda la penalty totale a max 16 (era 30). Riduce varianza Q-target nel buffer DDQN. Cliff meno aspro vicino a collision senza perdere segnale (gradient quadratic resta).

**Predictions verificabili post-training:**
- Avg reward M1 (last 30 ep) > 1500 (era -898 in Round 1).
- Max-steps survival rate M1 training > 25% (era 11.4%).
- M1 test success rate > 30% (era 0%).
- M2 test success rate ≥ 75% (era 82%, accettiamo regressione lieve).

Se H1 corretta ma H2 dannosa: M1 success migliorato ma M2 crollato (>10 punti). Discriminazione possibile.

## 4. Literature Support

- **Pfeiffer et al. 2017** ("From Perception to Decision", IROS): LIDAR navigation con front sector narrow ±30°. Identico al nostro fix.
- **Henderson et al. 2018** ("Deep RL that Matters", AAAI): bounded shaping reduces inter-seed variance per DDQN.
- **Long et al. 2018** (ICRA): quadratic obstacle penalty robust in narrow corridors → giustifica conservare shape.
- **Cobbe et al. 2019** (ICML): variable isolation in ablation studies → giustifica singola modifica per round.
- **Mnih et al. 2015** (Nature): reward magnitude bounding aids DDQN stability.

## 5. Changes

### 5.1 Code changes — `src/my_usv/scripts/usv_logic.py`

**Current (riga 24-26):**
```python
right_dist = float(np.min(scan[0:15]))
front_dist = float(np.min(scan[15:35]))
left_dist  = float(np.min(scan[35:50]))
```

**New:**
```python
right_dist = float(np.min(scan[0:20]))    # 108° destra (era 81°)
front_dist = float(np.min(scan[20:30]))   # 54° centro (era 108°)
left_dist  = float(np.min(scan[30:50]))   # 108° sinistra (era 81°)
```

**Current (riga 42, 46, 50):**
```python
danger_penalty += 20.0 * (severity ** 2)  # front
danger_penalty += 5.0 * (severity ** 2)   # side right
danger_penalty += 5.0 * (severity ** 2)   # side left
```

**New:**
```python
danger_penalty += 10.0 * (severity ** 2)  # front (era 20)
danger_penalty += 3.0 * (severity ** 2)   # side right (era 5)
danger_penalty += 3.0 * (severity ** 2)   # side left (era 5)
```

Nessun altro cambio. `SPACE_BONUS_WEIGHT`, `FRONT_DANGER`, `SIDE_DANGER`, `COLLISION_DIST` invariati.

### 5.2 Test changes — `src/my_usv/test/test_usv_logic.py`

**Test esistenti che restano invariati e devono continuare a passare** (uniform scan, no danger trigger):
- `test_collision_returns_minus_1000_and_done`
- `test_collision_triggered_by_single_ray_below_threshold`
- `test_reward_at_exact_collision_boundary`
- Tutti i test `process_lidar` (sector indipendente da reward).

**Test esistenti pre-broken (merge16_05 augmented reward vs Feng pure +5/-1000):**
- `test_no_collision_returns_exactly_5`
- `test_action_index_does_not_affect_reward`
- `test_obstacle_far_away_no_collision`

Questi falliscono già su Round 1 (code returns ~7.0 con space_bonus 2.0, non 5.0). **Fuori scope Round 2.** Non toccare. Flag in `BRIEFING_20_05.md` per Round 3 fix.

**Test nuovi da aggiungere (3):**

1. `test_front_sector_narrow_indices_20_30`: input scan con close obstacle solo in bin 15 (era front, ora right). Verifica che `compute_reward` non triggera `front_danger` (vale a dire: passa il check tramite `front_dist = min(scan[20:30])`, non `[15:35]`).
   ```python
   def test_front_sector_narrow_indices_20_30():
       scan = _clear_scan()
       scan[15] = 0.5  # close obstacle in bin 15 (right area, not front)
       reward, done = compute_reward(scan, action_index=5)
       # front_dist = min(scan[20:30]) = 5.0 → no front penalty
       # right_dist = min(scan[0:20]) = 0.5 → side penalty triggers
       # severity = (0.45 - 0.5)/(...) → negativo, no trigger (0.5 > 0.45)
       # quindi reward = 5 + bonus, no danger
       assert done is False
       assert reward > 4.0  # solo bonus + base, no penalty
   ```

2. `test_front_penalty_max_weight_is_10`: input scan con front bin appena sopra collision → verifica penalty front bounded ≤ 10.
   ```python
   def test_front_penalty_max_weight_is_10():
       scan = _clear_scan()
       # bin centrali 20-30 = front. Set tutti a 0.26 (just above collision 0.25)
       scan[20:30] = 0.26
       reward, done = compute_reward(scan, action_index=5)
       assert done is False
       # severity ≈ (1.5 - 0.26) / (1.5 - 0.25) ≈ 0.992
       # penalty front ≈ 10 * 0.992^2 ≈ 9.84
       # reward = 5 + bonus - steering - 9.84
       # bonus ≈ 2.0 * mean ≈ 2.0 (mean dominato da 5.0)
       # reward ≈ -2.84
       assert -4.0 < reward < -1.0  # range bounded
   ```

3. `test_side_penalty_max_weight_is_3`: input scan con side bin appena sopra collision → penalty side ≤ 3.
   ```python
   def test_side_penalty_max_weight_is_3():
       scan = _clear_scan()
       # bin 0 = right side. Set a 0.26
       scan[0] = 0.26
       reward, done = compute_reward(scan, action_index=5)
       assert done is False
       # severity ≈ (0.45 - 0.26) / (0.45 - 0.25) ≈ 0.95
       # penalty side ≈ 3 * 0.95^2 ≈ 2.71
       # reward ≈ 5 + 2 - 2.71 ≈ 4.29
       assert reward > 3.5  # base reward dominante, penalty bounded
   ```

### 5.3 No changes elsewhere

- `usv_env.py`: invariato (sector split usato solo in `compute_reward`)
- `train.py`, `train_core.py`, `test.py`, `ddqn_model.py`: invariati
- `start_train_multimaze.sh`: invariato (M1+M2 ratio 1:2 mantenuto)
- Spawn lists: invariate

## 6. Training plan

- **Reset checkpoint** (training from scratch — confronto pulito vs Round 1).
- **5000 episodi totali**, BLOCK_PATTERN=(1,2,2), 100 ep/block, identico Round 1.
- ε decay 0.999 → min 0.05, identico.
- Tempo stimato: ~8 h wall-clock (GAZEBO_SPEED=4, in linea con Round 1).
- Log: `training_log.csv` standard.
- Best model selection: `avg100` globale (invariato — l'analisi del bias best-model è separata, va in Round 3).

## 7. Testing plan

Identico Round 1:
- 90 episodi × 3 mazes (M1, M2, M3) = 270 ep totali, con ε=0.
- Output `test_results.csv` standard (271 righe: header + 270 test).

## 8. Evaluation

Post-training, analisi con `analysis_multi_maze_v_19_05.py` (riusabile diretta). Confronto Round 1 vs Round 2 su 4 metriche:

| Metric | R1 baseline | R2 target | R2 fail condition |
|---|---|---|---|
| M1 test success rate | 0% | ≥30% | <10% |
| M1 avg reward last 30 ep | -898 | >+1500 | <0 |
| M2 test success rate | 82% | ≥75% | <70% |
| M3 test success rate | 0% | ≥0% | regression |

**Decision tree post-Round 2:**
- M1 ≥30% & M2 ≥75% → R-α confermato. Procedere a Round 3 (analizzare M3 / best-model bias).
- M1 ≥30% & M2 <70% → weights troppo ridotti, rebound. Considerare R-α' con weights 15/4 (middle ground).
- M1 <10% → R-α insufficient. H1 falsificata o solo parzialmente. Considerare R-β (linearize) o investigare ulteriore.
- Tutto regredito → bug. Revert.

## 9. Risks

| Risk | Severità | Mitigation |
|---|---|---|
| M2 regression >10 punti | Medio | Test discriminante (#8). Revert se confermato. |
| Weights ridotti → segnale debole vicino collision → più crash | Medio-Basso | Quadratic preserva gradient force a small d. Max penalty 10 ancora > base reward 5 → robot fugge muri. |
| Reset checkpoint perde 5000 ep di learning | Basso | Reset necessario per confronto pulito. Round 1 baseline conservato in ANALISI_19_05. |
| Side bins ora 108° → side_danger triggera su walls front-laterali | Basso | Side threshold 0.45m molto vicino. Trigger solo con wall realmente vicino. M2 corridoi 2.5m → no trigger. |

## 10. Rollback

Branch isolato `ddqn_en_20_05`. Rollback = `git checkout ddqn_en_19_05`. Round 1 checkpoint+results preservati in ANALISI_19_05/.

## 11. Acceptance criteria

- [ ] `usv_logic.py` modificato secondo §5.1
- [ ] 3 nuovi test in `test_usv_logic.py` passano
- [ ] Test esistenti che dipendono da weights/sector aggiornati e passano
- [ ] Training 5000 ep completato, log integro
- [ ] Test 90 ep × 3 mazes (270 totali) completato
- [ ] `analysis_multi_maze_v_19_05.py` eseguito, plot generati
- [ ] Confronto numerico R1 vs R2 in `ANALISI_20_05/comparison.md`
- [ ] Decision tree §8 applicato, prossimo round identificato

## 12. Open questions

Nessuna. Spec self-contained.

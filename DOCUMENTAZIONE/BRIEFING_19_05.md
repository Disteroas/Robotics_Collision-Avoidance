# Briefing 2026-05-19 — Analisi `ddqn_enhanced_18_05` + Design Round 1

**Branch precedente:** `ddqn_enhanced_18_05` (training completato 2026-05-19)
**Branch in design:** `ddqn_round1_19_05`
**Autori:** Davide Covolo (+ Claude)

---

## 1. Risultati `ddqn_enhanced_18_05` — sintesi

Training 5000 ep, M2-only, frame stack k=3 + heading [cos,sin] da odom topic.

| Metrica | Valore | vs merge16 run1 baseline |
|---|---|---|
| Final avg100 | 742.3 | +545 pts |
| Best avg100 | 875.1 @ ep 4962 | +521 pts |
| Crash all | 80.7% | -0.1pp |
| Crash last 100 | 60.0% | uguale |
| M1 test | 0% | n/d → regressione vs merge12 (66.7%) |
| **M2 test** | **51%** | **+5pp record storico** |
| M3 test | 0% | costante su 9 esperimenti |

**Test M2 spawn-by-spawn (90 ep totali):**
- F1, A1, F3: **100%** ognuno (45/45 totali) — frame stacking + heading hanno sbloccato A1 e F3
- F2, C2, D1: **0%** ognuno (0/45) — distribuzione bimodale, fallimento strutturale

**Curva reward training (vedi `ANALISI_18_05/plots/01_reward_curve.png`):**
- avg100 > 0 raggiunto a ep 2658 (più tardi vs merge16, ma valore più alto)
- crescita monotona dopo ep 3000 → no instabilità
- best avg100 a fine training (ep 4962) → curva ancora in salita, eventuale beneficio di episodi extra

**Crash rate (vedi `ANALISI_18_05/plots/03_crash_rate.png`):**
- plateau ~60% da ep 4000 in poi → asintoto confermato

---

## 2. Diagnosi successi e fallimenti

### Cosa ha funzionato

**A. Frame stacking k=3 (Mnih 2015):**
A1 e F3 da spawn problematici (training max-steps 14% e 29% in merge16) a 100% test in ddqn_enh. La rete ora distingue stati simili tramite contesto temporale, traiettoria deducibile dal delta-LIDAR fra t, t-1, t-2.

**B. Heading [cos(yaw), sin(yaw)] da odom (Mirowski 2016):**
Indirettamente confermato: A1 (yaw=0°) e F3 (yaw=180°) hanno comportamenti opposti corretti. Robot capisce in quale direzione sta puntando.

**C. Min-pooling LIDAR (preservato da merge16):**
Nessuna regressione. Pixel più informativi rispetto a uniform sampling.

### Cosa NON ha funzionato

**X1. M1 = 0% (regressione catastrofica):**
M2-only training → policy iper-specializzata su corridoi stretti. In M1 (geometria più aperta) il robot fa wall-following aggressivo e finisce contro pareti distanti che non si aspetta.

**X2. M3 = 0% (costante su 9/10 esperimenti):**
Frame stacking risolve POMDP *dentro* M2, non *fuori*. La rete ha imparato pattern temporali specifici della geometria di M2 → in M3 questi pattern non hanno significato.

**X3. F2, C2, D1 = 0% in test (e in training):**
Bimodalità totale: 3 spawn perfetti, 3 spawn falliti. No middle ground → non è rumore esplorativo, è limite strutturale della policy. Frame stack + heading aiutano dove c'è diversità sensoriale; falliscono dove l'ambiguità è geometrica.

**X4. Heading channel sotto-pesato (1.3%):**
2/152 dim sotto Xavier init = pesi `fc1.weight[:, 150:152]` sotto-aggiornati. Predizione del briefing §12.7 confermata indirettamente: dove heading è discriminante (A1 vs F3 yaw opposti) funziona, dove heading è ambiguo (F2/C2 yaw simili) non aiuta.

**X5. D1 spawn unfair:**
Analisi geometrica offline (vedi `analysis/maze2_geom_check.py`): clearance 0.512m OK ma heading W spinge robot in muro centrale entro 60 step. Crash deterministico, non legato a policy.

---

## 3. Storico comparato (10 esperimenti)

```
M2 test:  0%  → 13% → 33% → 40% → 46% → 51%
          paper  m15   m16r2  m17  m16r1  ddqn_enh

M3 test:  SEMPRE 0% (eccetto merge14 brief: 13%, single run)

Crash last 100: 82% → 67% → 60% (plateau)
```

**Pattern strutturale:** asintoto M2 ~ 50% e M3 = 0% con setup attuale. Nessun incremento monotono ulteriore aspettato da fix incrementali algoritmici (n-step, Dueling). Per superare asintoto: cambio di **training distribution** (multi-env) o cambio di **algoritmo** (PPO/SAC stocastici).

---

## 4. Decisioni brainstorming 2026-05-19

### 4.1 Budget runs
2 training run in 2 settimane Track A. Round 1 + Round 2 (decisione data-driven).

### 4.2 Maze composition Round 1
**M1 + M2 training, ratio 1:2. M3 zero-shot test.**

Motivazione:
- M1 più semplice (2 spawn validati, geometria aperta) → easy wins precoci → buffer più sano
- Cobbe et al. 2019: multi-env training necessario per generalization
- M3 zero-shot = metrica scientifica pulita per la tesi
- Ratio 1:2 mantiene M2 dominante (51% baseline, target ≥ 45%)

### 4.3 Spawn D1 — relocation
Analisi geometrica ha rivelato che D1 non è geometricamente unfair (clearance 0.512m > soglia 0.40m). Problema = **heading W** da posizione (1.5, 0.0):
- Robot facing W, v=0.5 m/s
- 3 metri verso x decrescente = muro centrale del labirinto
- Tempo a impatto: 3 / 0.5 = 6 s = 60 step
- Match preciso con dato training: D1 avg 56 step

**Decisione:** ricolloca D1 → **(3.5, -0.5, yaw=π/2 N)**. Clearance 0.959m, corridoio aperto a nord. Mantiene zone D coverage.

**Top 4 candidati esaminati (vedi `analysis/plots/maze2_spawn_map_v2.png`):**

| Pos | Clearance |
|---|---|
| **(3.50, -0.50)** ← scelto | 0.959m |
| (2.50, -2.50) | 0.933m |
| (0.75, -0.50) | 0.914m |
| (3.75, +1.50) | 0.907m |

### 4.4 Approccio Round 1 — Conservativo (A)
Bundle 3 cambiamenti complementari:
1. Multi-maze M1+M2 (ratio 1:2)
2. Domain randomization LIDAR (σ=0.02, training-only)
3. Spawn D1 ricollocato

Heading × 10 replication, n-step returns, Dueling DDQN → **riservati a Round 2** in base a outcome Round 1.

Motivazione isolation: feedback memory "Tenere variabili isolate tra esperimenti". Multi-maze + DR sono concettualmente coerenti (entrambi = data diversity). Heading × 10 è ortogonale.

### 4.5 Track B (PPO) — sospeso
Decisione utente: rivalutazione fra qualche giorno in base a esito Round 1. Track A consuma tutte e 4 le persone per ora.

---

## 5. Architettura Round 1

### State vector (invariato da ddqn_enh)
```
STATE_DIM = 152
  = 50 (LIDAR_BEAMS) × 3 (FRAME_STACK)   = 150
  + 2 ([cos(yaw), sin(yaw)] da /odom)    =   2
```

### Reward (invariato da merge16/ddqn_enh)
```python
# usv_logic.py compute_reward()
+5.0 base
+ SPACE_BONUS_WEIGHT * mean(scan) / LIDAR_MAX_RANGE       # 0..2.0
- 0.02 * abs(action - 5)                                  # steering, 0..0.10
- 20 * severity²  if front_dist < 1.5m                    # quadratic, 0..20
- 5  * severity²  if right_dist < 0.45m                   # quadratic, 0..5
- 5  * severity²  if left_dist  < 0.45m                   # quadratic, 0..5
−1000  se min(any) < 0.25m  → done
```

### Cambiamenti vs ddqn_enh

**1. `start_train_multimaze.sh` — BLOCK_PATTERN multi-maze:**
```bash
# Era:
BLOCK_PATTERN=(2)   # solo M2

# Round 1:
BLOCK_PATTERN=(1 2 2)   # M1, M2, M2 → ratio 1:2
```

**2. `usv_env.py` — DR LIDAR training-only:**
```python
# Nuovo: noise injection prima di compute_reward
# (solo durante training, NON in test)
DR_NOISE_STD = 0.02  # 10cm su range 5m

def step_action(self, action_index: int, training: bool = True):
    cmd = Twist(); ...
    self._wait_sim_seconds(STEP_DT)
    rclpy.spin_once(self, timeout_sec=0.05)

    scan_clean = self.current_scan.copy()
    if training:
        scan_noisy = scan_clean + np.random.normal(0, DR_NOISE_STD, LIDAR_BEAMS)
        scan_noisy = np.clip(scan_noisy, 0.0, LIDAR_MAX_RANGE)
        reward, done = compute_reward(scan_clean, action_index)
        self.current_scan = scan_noisy.astype(np.float32)  # state usa noisy
    else:
        reward, done = compute_reward(scan_clean, action_index)
    self._push_frame()
    return self.get_state(), reward, done
```

**Nota implementazione:** reward su scan pulito (per non disturbare gradiente), state su scan noisy (per robustezza). Standard DR pattern.

**3. `usv_env.py` — spawn D1:**
```python
SPAWN_LISTS[2] = [
    (-6.0,  0.0,  0.0  ),  # A1
    (-7.0,  5.0,  0.0  ),  # C2
    ( 3.5, -0.5,  1.571),  # D1_NEW (era 1.5, 0.0, 3.142)
    (-4.5, -3.5,  0.0  ),  # F1
    (-1.5, -4.0,  1.571),  # F2
    ( 6.0,  6.0,  3.142),  # F3
]
TEST_SPAWN_LISTS[2] = SPAWN_LISTS[2]  # invariato pattern
```

---

## 6. Target Round 1

| Metrica | Baseline ddqn_enh | Target | Stretch | Failsafe |
|---|---|---|---|---|
| M2 success | 51% | ≥ 45% | ≥ 55% | < 40% = abort |
| M1 success | 0% | ≥ 50% | ≥ 70% | < 30% = M1 problem |
| **M3 success** | **0%** | **> 0%** | ≥ 15% | = 0% → Round 2 escalation |
| Crash last 100 | 60% | ≤ 60% | ≤ 50% | > 70% = instabile |

**Criterio go/no-go Round 2:**
- M3 ≥ 5% → ottimizza con heading × 10 in Round 2
- M3 = 0% → escalation: PPO Track B parallelo, o ricorrenza LSTM Track A

---

## 7. Letteratura di supporto

- **Cobbe et al. 2019** — "Quantifying Generalization in Reinforcement Learning": multi-environment training necessario per generalization
- **Tobin et al. 2017** — "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
- **Peng et al. 2018** — "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"
- **Mnih et al. 2015** — DQN: frame stacking k=4 (qui k=3)
- **Mirowski et al. 2016** — "Learning to Navigate in Complex Environments": heading necessario per disambiguazione
- **Feng 2021** — paper di riferimento USV navigation

---

## 8. Limitazioni note Round 1

- Spawn list non ri-validata empiricamente con `validate_spawn.py` runtime (sostituita da analisi geometrica offline)
- DR LIDAR σ=0.02 = scelta a priori (potrebbe richiedere tuning fra round)
- Heading channel ancora 1.3% input (deliberatamente non risolto in Round 1 per isolation)
- Single seed (no varianza ignota)

---

## 9. File coinvolti

| File | Modifica | Costo |
|---|---|---|
| `usv_env.py` | DR injection in `step_action()` + spawn D1 nuova | 30 min |
| `start_train_multimaze.sh` | `BLOCK_PATTERN=(1 2 2)` | 5 min |
| `analysis/maze2_geom_check.py` | Analisi geometrica (creato 2026-05-19) | done |
| `train.py` / `test.py` | Param `training=True/False` su step_action | 15 min |
| Test suite | Aggiornare per spawn nuovo + DR pattern | 1h |

---

## 10. Next steps

1. Approvazione design completo (Sezione 2-3 brainstorming TBD)
2. Spec scritto in `docs/superpowers/specs/2026-05-19-ddqn-round1-design.md`
3. Implementation plan in `docs/superpowers/plans/2026-05-19-ddqn-round1.md`
4. Esecuzione via subagent-driven-development
5. Training 5000 ep + test 90 ep/maze
6. Analisi risultati → decisione Round 2 (heading × 10 o escalation)

---

*Generato: 2026-05-19 | Branch design: ddqn_round1_19_05*

# Design — Replica fedele di Feng 2021 sul branch `paper_metric_feng`

**Data:** 2026-05-25 · **Branch:** `paper_metric_feng` (creato da `paper_metric_base @ c5b8a06`)
**Obiettivo:** rendere l'**agente** (reward, stato, rete, iperparametri, protocollo di training) **identico** a Feng et al. 2021, mantenendo solo la **nostra policy di valutazione**. È l'esperimento #1 del piano a 4 ("Feng fedele — baseline letteratura, àncora varianza").

**Fonte:** Feng, S.; Sebastian, B.; Ben-Tzvi, P. *A Collision Avoidance Method Based on Deep Reinforcement Learning.* Robotics 2021, 10, 73. MDPI. doi:10.3390/robotics10020073.

---

## 1. Principio di fedeltà

"Identico a Feng" si applica all'**agente e al protocollo di training**: rappresentazione dello stato, spazio azioni, reward, rete, loss, target update, ε-policy, numero di episodi, mappa di training.

**Unica eccezione voluta:** la **policy di valutazione** resta la nostra (round-robin sugli spawn, ε=0 greedy, success@500 step, su M1/M2/M3, `run_meta.json`, `aggregate_seeds.py`). Motivo: la valutazione è lo strumento di misura onesto del progetto, non un parametro di Feng. Feng valutava "n. collisioni in 5 minuti continui" — metrica che abbiamo già giudicato debole.

L'**ambiente fisico** resta il nostro (USV in Gazebo, mazes M1/M2/M3, kinematics a v lineare fissa). Portiamo il *metodo* di Feng sul nostro env, non ricostruiamo il robot STORM (Feng è su robot terrestre skid-steer, non USV).

**Ruolo scientifico (cornice — leggere prima di interpretare i risultati).** Questo branch è una **baseline / àncora di varianza**, **NON** una prova che "Feng non funziona". Un eventuale fallimento della replica **non** è attribuibile al metodo di Feng: spec incompleta riempita da noi (~8 iperparametri) + env/metrica/mappe diversi sono confondenti. Il valore è **comparativo**: misurare le alternative (r_alpha…) contro questa baseline **sotto protocollo identico**. Tesi e guardrail completi in [`../../../DOCUMENTAZIONE/PAPER_ANALYSIS/letteratura_drl_collision_avoidance.md`](../../../DOCUMENTAZIONE/PAPER_ANALYSIS/letteratura_drl_collision_avoidance.md) §0-bis.

---

## 2. Parametri Feng — catalogo completo

### 2.1 Specificati nel paper (da replicare)

| Parametro | Valore Feng | §/Eq |
|---|---|---|
| Algoritmo | DDQN (PER testato e scartato) | §3.2 |
| Hidden layers | 300 → 300, ReLU | §3.4 |
| Output | 11 Q-values | §3.4 |
| Stato | 50 range LIDAR, da 512 ray (270°), **selezionati uniformemente**, clip **[0, 5.0 m]**, `st = Ot` | §5.1, §3.4 |
| Frame stacking | **nessuno** (stato = singola osservazione) | §3.4 |
| Azioni | 11, stessa v lineare, ω = −0.8 + 0.16·m, m=0..10 | §3.4 |
| Reward | **+5** (no collisione) / **−1000** (collisione), puro | Eq.4 |
| Loss | **MSE** (mini-batch GD) | Eq.5 |
| Target terminale | y = r se collisione (no bootstrap); altrimenti y = r + γ·Q(s', argmax) | Eq.6, Alg.1 |
| Target update | hard, θ⁻←θ ogni N step | Alg.1 r.16 |
| ε-greedy | start 1.0, floor 0.05, εₖ₊₁ = β·εₖ | §3.3, §5.2 |
| β decay | **0.999** (migliore dei tre testati 0.997/0.998/0.999) | §5.2, §5.5 |
| Episodi | **3000** | §3.4, §5.2 |
| Control rate | 10 Hz | §5.5 |
| Mappa training | **una sola** mappa complessa (Map 2), reset a **posizione random** | §5.2, §3.4 |
| Grad clipping | **non menzionato** (→ assente) | — |
| Domain randomization | **non menzionato** (→ assente) | — |

### 2.2 NON specificati da Feng (teniamo i valori r_alpha, dichiarati non-da-Feng)

learning rate · optimizer · γ (discount) · batch size · replay buffer size · N (intervallo target update) · v lineare · max-steps T per episodio.

→ Valori adottati: γ=0.99 · Adam lr=0.00025 · batch 64 · buffer 100k · N=5000 step · v=0.5 m/s · max-steps 500.
Questi **non** provengono da Feng (il paper è muto) e vanno dichiarati come tali in ogni report.

---

## 3. Stato attuale del codice (da `paper_metric_base`)

| Componente | Attuale | Feng | Azione |
|---|---|---|---|
| `STATE_DIM` (ddqn_model.py) | 152 (50×3 + heading 2) | 50 | **cambia** |
| Frame stack (usv_env) | 3 | 1 | **cambia** |
| Heading cos/sin (get_state) | sì | no | **rimuovi** |
| Normalizzazione net-input (_push_frame) | `/5 → [0,1]` | [0,5] (clip, no norm) | **rimuovi /5** |
| Domain rand. noise (step_action) | σ=0.02 | nessuna | **rimuovi** |
| LIDAR→50 (process_lidar) | min-pool (array_split + min) | selezione uniforme | **cambia** |
| Reward (compute_reward) | +5 + space_bonus − steering − front/side / −1000 | +5 / −1000 | **sostituisci** |
| Grad clip (train_core) | clip_grad_norm 10.0 | nessuno | **rimuovi** |
| Loss | MSE | MSE | ok |
| Rete hidden/azioni/ε/β | 300×300×11 / ω=−0.8+0.16·idx / 1.0→0.05 / 0.999 | idem | **già identici** |
| Episodi (train.py) | 5000 | 3000 | **cambia** |
| Training maze | multimaze M1+M2 (BLOCK_PATTERN 1,2,2) | solo Map2 | **solo M2** |

---

## 4. Modifiche al codice (hardcode sul branch)

### 4.1 `src/my_usv/scripts/ddqn_model.py`
- `STATE_DIM = 152` → `STATE_DIM = 50`. Aggiornare commento (`# 50 LIDAR, Feng 2021: st=Ot, no frame-stack, no heading`).

### 4.2 `src/my_usv/scripts/usv_logic.py`
- **`process_lidar`**: sostituire il min-pool con **selezione uniforme** di 50 indici equispaziati sui ray grezzi, poi clip [0, 5.0]:
  ```python
  scan = np.nan_to_num(np.array(raw_ranges, np.float32), nan=max_range, posinf=max_range, neginf=max_range)
  scan = np.clip(scan, 0.0, max_range)
  idx = np.linspace(0, len(scan) - 1, n_bins).round().astype(int)
  return scan[idx]
  ```
- **`compute_reward`**: sostituire l'intera reward shaped con la reward pura di Feng (Eq.4):
  ```python
  def compute_reward(scan, action_index):
      if float(np.min(scan)) < COLLISION_DIST:
          return -1000.0, True
      return 5.0, False
  ```
  - `COLLISION_DIST = 0.25` resta (operazionalizzazione nostra della "collisione fisica" di Feng).
  - Le costanti di shaping (FRONT_DANGER, SIDE_DANGER, SPACE_BONUS_WEIGHT) e le slice settore **restano definite** perché `sector_distances`/`crash_sector` sono usate dal logging/eval (non dalla reward). `round_robin_spawn` resta (eval).

### 4.3 `src/my_usv/scripts/usv_env.py`
- `FRAME_STACK = 3` → `FRAME_STACK = 1`.
- **`_push_frame`**: rimuovere la normalizzazione `/LIDAR_MAX_RANGE` → il buffer contiene lo scan in metri [0,5].
- **`get_state`**: ritornare il singolo scan (50 dim), **senza** concatenazione frame e **senza** heading:
  ```python
  def get_state(self):
      if not self._frame_buffer:
          raise RuntimeError("get_state() prima di reset_environment()")
      return self._frame_buffer[-1].copy()   # 50 dim, [0,5] m
  ```
- **`step_action`**: rimuovere il blocco domain-randomization (`if training: noise = …`); `scan_for_state = self.current_scan` sempre.

### 4.4 `src/my_usv/scripts/train_core.py`
- Rimuovere `torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)`.
- Invariati: GAMMA=0.99, LR=0.00025 (Adam), BATCH_SIZE=64, MEMORY_CAPACITY=100k, TARGET_UPDATE_STEPS=5000, BETA_DECAY=0.999, EPSILON_START=1.0, EPSILON_MIN=0.05, MSELoss, target Double-DQN con `(1-d)` (= Feng Eq.6).

### 4.5 `src/my_usv/scripts/train.py`
- Episodi totali 5000 → **3000**.

### 4.6 Orchestrazione training M2-only
- Lo spawn random è **già nativo** in training (`random.choice(SPAWN_LISTS[maze_id])`, usv_env r.157). Serve solo eliminare l'interleaving M1+M2.
- Creare `start_train_feng.sh` (adattato da `start_train_multimaze.sh`): gestione lifecycle Gazebo identica, ma **sempre `--maze-id 2`** per tutti i blocchi fino a 3000 ep, mondo `labirinto_9b.world`. Niente BLOCK_PATTERN.
- Artefatti in `runs/feng/seed_<S>/` (via `--config feng`, che resta etichetta di path/seed).

### 4.7 Valutazione — INVARIATA
`test.py`, round-robin, ε=0, success@500, M1/M2/M3, `run_meta.json`, `aggregate_seeds.py`: nessuna modifica. M1 e M3 diventano automaticamente **held-out** per l'agente Feng (allenato solo su M2).

---

## 5. Divergenze residue inevitabili (da dichiarare)

1. **Spawn training:** Feng usa "posizione random qualsiasi" nel mondo; noi usiamo `random.choice` su un set curato di 6 spawn validati di M2 (vincolo del nostro env). Divergenza necessaria, documentata.
2. **Normalizzazione input:** Feng dice "preprocessed to 0–5 m" (clipping); non specifica se normalizza per la rete. Lettura letterale → feed [0,5]. ⚠️ Cambia la scala input mentre lr resta il nostro (Feng-muto) → **watch-item** di stabilità.
3. **Grad clip rimosso:** MSE + spike −1000 senza clip può divergere. ⚠️ **Rischio #1**: se il training a 3000 ep esplode (Q→∞, loss NaN), questo è il primo sospetto; il target terminale `y=r` (no bootstrap) mitiga ma non elimina. Facile ri-aggiungere se necessario.
4. **Budget vs r_alpha:** Feng 3000 ep, r_alpha 5000. Il confronto Feng-vs-r_alpha mescola **metodo + budget** → non è ablazione a variabile singola. Da dichiarare in ogni confronto.
5. **Dominio:** robot/env nostri (USV, M2≈"Map 2" di Feng), non STORM skid-steer.

---

## 6. Validazione

- **Smoke test (manuale, container):** `STATE_DIM==50`; `env.get_state().shape == (50,)`; valori stato in [0,5]; `compute_reward` ritorna solo `5.0` o `−1000.0`; nessun NaN nel primo blocco di training.
- **Unit test `test_usv_logic.py`:** i 4 test rotti erano scritti per la reward Feng pura (+5/−1000) → con questa modifica **devono tornare verdi**. Verificare. Aggiornare gli altri test che assumono lo stato 152 o la reward shaped.
- **Sanity training:** primo blocco (200 ep) senza divergenza (loss finita, ε decade come `0.999^k`).

---

## 7. Fuori scope

- MSI / cross-hardware.
- Lavoro `--max-steps` / `collision_free_steps`.
- Qualunque modifica a `paper_metric_base` (r_alpha resta lì, intatto).
- Fix del `*LIDAR_MAX_RANGE` vestigiale alla riga 169 di usv_env (latente, non legato a Feng).
- Esperimenti #2/#3/#4 del piano.

---

## 8. Risultato atteso

Un branch `paper_metric_feng` in cui l'agente DDQN è la replica fedele di Feng 2021 (stato 50 [0,5], reward +5/−1000, no frame-stack/heading/DR/grad-clip, 3000 ep, training random su M2), valutato con la nostra metrica rigorosa multi-seed su M1/M2/M3. Serve da **baseline di letteratura** e da àncora di varianza per il confronto con `r_alpha`.

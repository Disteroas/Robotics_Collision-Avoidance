# Base per simulazioni rigorose — Design

**Data:** 2026-05-22
**Branch:** `ddqn_en_20_05` (o nuovo branch dedicato `rigorous-base`)
**Scopo:** Costruire l'infrastruttura di misura validata da letteratura che renda le run **riproducibili, valutabili e diagnosticabili** — prerequisito a qualsiasi confronto futuro (DDQN onesto vs TD3). Questa è la "Fase 0 / spina dorsale" del piano in `ANALISI_STRATEGICA_2026-05-21.md`, e costituisce la Parte 2 (riproducibilità) del paper.

---

## 1. Obiettivo e contesto

Il Round 2 ha rivelato che due run con codice identico divergono enormemente (M1 0↔50%, M3 0↔23%) perché **nessun seed è fissato** e il logging è troppo povero per capire *come* l'agente fallisce. Senza questa base, ogni confronto tra round o tra algoritmi è n=1 → rumore letto come segnale (Henderson 2018; Agarwal 2021).

La base poggia su **3 pilastri**:
1. **Riproducibilità** — seed control.
2. **Valutazione** — protocollo deterministico + aggregazione multi-seed con statistica corretta.
3. **Osservabilità** — logging per-step in eval per diagnosticare e giudicare la policy.

## 2. Non-goal (confini di scope)

Questa fase **non** modifica nulla che cambi il comportamento dell'agente:
- NO modifiche a reward (`usv_logic.py`), rete (`ddqn_model.py`), iperparametri, action-space.
- NO action-space con decelerazione (round successivo, fix F1).
- NO reward direzionale (round successivo).
- NO refactor dell'agente per TD3 (l'eval è reso algorithm-agnostic, ma l'astrazione dell'agente è fuori scope).

Una variabile alla volta: prima la base di misura, poi gli esperimenti.

---

## 3. Pilastro 1 — Riproducibilità (seed control)

### 3.1 Nuovo modulo `src/my_usv/scripts/seeding.py`

```python
"""Controllo centralizzato del seed. Nessuna dipendenza ROS."""
import os
import random
import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Fissa tutte le sorgenti di randomness controllabili.

    NOTA: Gazebo (fisica + timing ROS) resta non deterministico.
    Il seed NON dà riproducibilità bit-a-bit, ma rende la varianza
    attribuibile e misurabile (Henderson 2018).
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
```

`torch.use_deterministic_algorithms(True)` **non** viene attivato: su CPU dà overhead e qui la non-determinismo dominante è Gazebo, non i kernel torch. Documentare questa scelta.

### 3.2 CLI

- `train.py` e `test.py`: aggiungere `--seed INT` (default `0`).
- `set_global_seed(args.seed)` chiamato come **primissima istruzione** di `main()`, prima di creare `UsvEnv`, `DDQNAgent`, o qualsiasi RNG.

### 3.3 Tracciamento del seed

- Salvato nel checkpoint (`save_ckpt` → nuova chiave `'seed'`).
- Scritto in un file `runs/<config>/seed_<S>/run_meta.json` con: seed, timestamp, git SHA, hostname (PC), config (reward version, maze pattern, episodi).

---

## 4. Pilastro 2 — Protocollo di valutazione

### 4.1 Round-robin deterministico

In `usv_env.reset_environment(maze_id, test_mode=True)`, sostituire `random.choice(spawn_list)` (`usv_env.py:146`) con una **selezione round-robin** quando `test_mode=True`:

- L'env tiene un contatore per-maze `self._test_spawn_idx`.
- Restituisce `spawn_list[idx % len(spawn_list)]`, poi incrementa.
- In training (`test_mode=False`) resta `random.choice` (ora seedato → riproducibile).

Numero spawn per maze (da `TEST_SPAWN_LISTS`):

| Maze | Spawn | n |
|---|---|---|
| M1 | P1(−2.9,−2.0), P2(1.0,−1.0) | 2 |
| M2 | A1, C2, D1, F1, F2, F3 | 6 |
| M3 | (−2.5,−0.25) | 1 |

**Formula episodi:** `episodes = n_spawns × reps`. Nuovo CLI `--reps INT` (default 30). Quindi M1=60, M2=180, M3=30 episodi. Ogni spawn ottiene esattamente `reps` episodi → coverage bilanciata e confrontabile tra seed e tra algoritmi.

> Cambio interfaccia: `test.py` non passa più `--episodes` ma `--reps`; il totale è derivato. Mantenere `--episodes` come override opzionale ignorato se `--reps` presente (retrocompatibilità script).

### 4.2 Criterio di successo (fissato e documentato)

`successo = episodio termina per raggiungimento di MAX_STEPS (500) senza collisione`. Invariato rispetto a oggi, ma reso esplicito nella docstring e nel `run_meta.json`. ε=0.0 greedy.

### 4.3 Confine algorithm-agnostic

Round-robin, logging e aggregazione dipendono solo da `stato → azione`. Per la Traccia B (TD3, azione continua) basterà sostituire il decoder d'azione; il resto della pipeline di eval è riusabile senza modifiche.

---

## 5. Pilastro 3 — Osservabilità (logging)

### 5.1 Interfaccia `last_info` in `usv_env`

`step_action()` popola `self.last_info` (dict) ad ogni step, così `test.py` non ricostruisce le distanze da `state*5.0`:

```python
self.last_info = {
    'front_dist': float,   # min settore frontale (bin [20:30])
    'left_dist':  float,   # min settore sinistro (bin [30:50])
    'right_dist': float,   # min settore destro  (bin [0:20])
    'min_lidar':  float,   # min globale
}
```

I confini settore replicano `usv_logic.py` (front [20:30], right [0:20], left [30:50]) — definirli come costanti condivise importate da `usv_logic` per non duplicare i numeri magici.

### 5.2 CSV eval per-step — `runs/<config>/seed_<S>/eval_steps_m<maze>.csv`

Colonne (default, riga "leggera"):

```
episode, step, spawn, action, q_chosen, q_max, q_spread,
front_dist, left_dist, right_dist, min_lidar, reward, done
```

- `q_chosen` = Q dell'azione scelta; `q_max` = max Q; `q_spread` = `q_max − q_min` (saturazione/confidenza della policy).
- Tutti i float arrotondati a 4 decimali.
- **Flag `--log-q-full`**: aggiunge 11 colonne `q0..q10` (vettore Q completo). Spento di default per non gonfiare i CSV.

**Stima dimensioni (eval-only):** ~60–180 ep/maze × ~250 step medi ≈ 15k–45k righe/maze, ~80 byte/riga → **~1–4 MB/maze/seed**, ~30–50 MB per config (3 maze × 5 seed). Trascurabile. Con `--log-q-full` ~2×. (Lo step-level in training sarebbe stato GB → escluso per design.)

### 5.3 Causa-crash

All'episodio terminato con collisione, scrivere una riga in `runs/<config>/seed_<S>/eval_crashes_m<maze>.csv`:

```
episode, spawn, crash_step, crash_sector, crash_dist, last_actions
```

- `crash_sector` ∈ {front, left, right} = settore con distanza minima allo step di crash.
- `last_actions` = ultime 5 azioni (stringa, es. `"5,5,6,5,4"`) → distingue P2 "va dritto" (azioni ~5) da "sterza nel muro".

### 5.4 Logging training (resta episode-level)

`training_log.csv` invariato + **1 colonna** `crash_sector` (vuota se non-crash). Nessun per-step in training.

---

## 6. Orchestrazione e layout file

### 6.1 Layout per-seed

```
runs/<config>/seed_<S>/
  checkpoint.pkl
  best_model.pth
  training_log.csv
  eval_steps_m1.csv  eval_steps_m2.csv  eval_steps_m3.csv
  eval_crashes_m1.csv ...
  eval_summary.csv          # 1 riga per (maze) con success rate, per questo seed
  run_meta.json
```

`<config>` = etichetta esperimento (es. `r_alpha`, `r_directional`). Default `default`.

### 6.2 Start script

- `start_train_multimaze.sh` e `start_test.sh`: accettano `--seed S` e `--config NAME`, passano a Python e scrivono in `runs/<config>/seed_<S>/`.
- **Assegnazione 5 seed ai 4 PC** (documentata nel README della cartella `runs/`): es. PC1→{1,5}, PC2→{2}, PC3→{3}, PC4→{4}. I seed girano in parallelo sulle macchine.

### 6.3 Backup automatico (guard anti-perdita)

Nel path di reset (`start_train_multimaze.sh --reset` o equivalente): **prima** di cancellare checkpoint/CSV, copiare `runs/<config>/` in `ANALISI_TRAINING/<data>/`. Se la copia fallisce, abortire il reset. (Lezione R1: CSV grezzi persi.)

### 6.4 `.gitignore`

- Aggiungere `runs/` (artefatti pesanti: checkpoint, CSV per-step).
- Versionare in git **solo** i CSV aggregati e i plot prodotti dal tool di §7, dentro `ANALISI_TRAINING/`.

---

## 7. Tool di aggregazione — `analisi_maze/aggregate_seeds.py`

### 7.1 Input/Output

- **Input:** `runs/<config>/seed_*/eval_summary.csv` (tutti i seed di una config) + opzionale `eval_steps_*` per statistiche fini.
- **Output:** `ANALISI_TRAINING/<data>/aggregate_<config>.csv` + plot, con per maze e per spawn:
  - **media ± deviazione standard** del success rate sui seed,
  - **IQM** (Inter-Quartile Mean, Agarwal 2021),
  - **intervallo di confidenza 95% via bootstrap** (stratified, ~10k resample),
  - n_seed usati.

### 7.2 Regola di reporting

Mai riportare il massimo tra i seed. Output sempre come `mean ± std (IQM; 95% CI)`. (Henderson 2018; Agarwal 2021).

---

## 8. Mappa delle modifiche per file

| File | Azione |
|---|---|
| `src/my_usv/scripts/seeding.py` | **Crea** — `set_global_seed` |
| `src/my_usv/scripts/train.py` | `--seed` + `set_global_seed` + `crash_sector` in CSV |
| `src/my_usv/scripts/test.py` | `--seed`, `--reps`, `--config`, `--log-q-full` + logging per-step/crash + `eval_summary.csv` |
| `src/my_usv/scripts/usv_env.py` | round-robin in `reset_environment(test_mode)`, contatore `_test_spawn_idx`, `last_info` in `step_action` |
| `src/my_usv/scripts/usv_logic.py` | esporre costanti confini settore (no duplicazione) |
| `src/my_usv/scripts/train_core.py` | `seed` in `save_ckpt`/`load_ckpt` |
| `analisi_maze/aggregate_seeds.py` | **Crea** — aggregazione multi-seed |
| `start_train_multimaze.sh` | `--seed`/`--config`, layout `runs/`, backup-guard su reset |
| `start_test.sh` | `--seed`/`--config`/`--reps`, layout `runs/` |
| `.gitignore` | aggiungi `runs/` |

---

## 9. Strategia di test

Test puri (no ROS/Gazebo), in `src/my_usv/scripts/` o `tests/`:

1. **`seeding`**: dopo `set_global_seed(42)`, due sequenze di `random.random()`, `np.random.rand()`, `torch.rand(1)` sono identiche a una seconda chiamata con stesso seed; diverse con seed diverso.
2. **Round-robin**: simulare 7 reset `test_mode` su M2 (6 spawn) → sequenza spawn = `[0,1,2,3,4,5,0]` (deterministica, copertura uniforme).
3. **`aggregate_seeds`**: dati 3 CSV summary fittizi con success {0.4, 0.5, 0.6} → mean=0.5, std calcolata, IQM e CI nel range atteso.
4. **Causa-crash**: episodio fittizio con front_dist minima → `crash_sector == 'front'`.

`usv_env`/`test.py` parti ROS: smoke test manuale (1 seed, `--reps 2`) dentro container, verificare che i CSV si popolino con le colonne attese.

---

## 10. Riferimenti

- Henderson et al., 2018 — *Deep RL that Matters* (varianza da seed, reporting). arXiv:1709.06560
- Agarwal et al., 2021 — *Statistical Precipice* / rliable (IQM, bootstrap CI). arXiv:2108.13264
- Cobbe et al., 2019 — generalizzazione, train≠test (motiva M3 zero-shot). arXiv:1812.02341

---

*Decisioni confermate dall'utente (2026-05-22): scope minimale + eval riusabile; per-step solo in eval, training episode-level + crash_sector; round-robin deterministico; 5 seed; layout `runs/<config>/seed_<S>/`; `runs/` in `.gitignore`; vettore Q completo dietro flag.*

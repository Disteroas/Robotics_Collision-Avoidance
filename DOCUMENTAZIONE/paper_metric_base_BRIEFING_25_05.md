# BRIEFING 25-05 — Branch `paper_metric_feng`: replica fedele di Feng 2021 (baseline)

**Branch:** `paper_metric_feng` (creato da `paper_metric_base @ c5b8a06`)
**Data:** 2026-05-25
**Per:** chiunque (utente + colleghi) debba lanciare il training della baseline Feng.
**Spec:** [`../docs/superpowers/specs/2026-05-25-feng-faithful-params-design.md`](../docs/superpowers/specs/2026-05-25-feng-faithful-params-design.md) · **Piano:** [`../docs/superpowers/plans/2026-05-25-feng-faithful-params.md`](../docs/superpowers/plans/2026-05-25-feng-faithful-params.md) · **Letteratura/tesi:** [`PAPER_ANALYSIS/letteratura_drl_collision_avoidance.md`](PAPER_ANALYSIS/letteratura_drl_collision_avoidance.md) §0-bis.

---

## 1. Cosa è stato fatto

Reimplementato l'**agente DDQN identico a Feng et al. 2021** (*A Collision Avoidance Method Based on Deep Reinforcement Learning*, Robotics 2021, 10, 73, MDPI, doi:10.3390/robotics10020073) sul nostro ambiente UGV/Gazebo. Tutto ciò che Feng specifica è stato replicato; l'unica eccezione voluta è la **policy di valutazione**, che resta la nostra (più rigorosa).

## 2. Perché — la baseline, non un "gotcha"

Questo branch serve da **baseline di letteratura / àncora di varianza**, **non** a "dimostrare che Feng non funziona". Distinzione cruciale (parere esperto DRL, concordato):

- **Affermiamo (difendibile):** Feng è **non riproducibile** (~8 iperparametri non specificati, niente codice, single-run) e **valutato debolmente** (metrica "collisioni in 5 minuti" single-run, favorevole). Reimplementato fedelmente e valutato sotto **protocollo rigoroso multi-seed su benchmark più severo**, i risultati sono molto più variabili/modesti dell'headline. Proponiamo alternative (r_alpha…) **sotto lo stesso protocollo**.
- **NON affermiamo:** *"Feng non funziona"*. Sarebbe confondente: (1) spec incompleta riempita da noi; (2) env diverso (USV vs robot terrestre STORM di Feng); (3) metrica più dura; (4) mappe diverse. Un eventuale fallimento della replica **non** è attribuibile al metodo di Feng.

Il valore è **comparativo**: misurare le alternative contro questa baseline, stesso protocollo. Contributo del progetto = **rigore + riproducibilità + ingegneria**.

## 3. Parametri (cosa è cambiato vs `r_alpha`)

### 3.1 Specificati da Feng → replicati
| Parametro | Valore |
|---|---|
| Stato | **50** LIDAR, selezione **uniforme** dai 512 ray, clip **[0, 5.0 m]**, `st = Ot` (no frame-stack, no heading, no normalizzazione /5) |
| Reward | **+5** per step / **−1000** alla collisione (puro, zero shaping) |
| Rete | 50 → 300 → 300 → 11, ReLU; loss **MSE** |
| ε-greedy | 1.0 → floor 0.05, decay β=**0.999** |
| Episodi | **3000** |
| Grad clipping | **assente** (Feng non lo menziona) |
| Domain randomization | **assente** |
| Training maze | **solo M2**, spawn random per-episodio (Feng allena su una sola mappa complessa) |

### 3.2 NON specificati da Feng → tenuti = `r_alpha` (DICHIARARE come non-da-Feng in ogni report)
γ=0.99 · Adam lr=0.00025 · batch 64 · replay buffer 100k · target update ogni 5000 step · v lineare 0.5 m/s · max-steps 500.
Feng è **muto** su questi: scelta nostra obbligata, va dichiarata.

### 3.3 Divergenze residue inevitabili (dichiarate)
1. Spawn training: noi `random.choice` su 6 spawn curati di M2 (Feng = posizione random qualsiasi).
2. Input [0,5]: lettura letterale di Feng ("preprocessed to 0–5 m"); non specifica normalizzazione rete.
3. Confronto Feng-vs-r_alpha mescola **metodo + budget** (Feng 3000 ep, r_alpha 5000) → non è ablazione a variabile singola.

## 4. Valutazione — INVARIATA (nostra)
`test.py`, round-robin sugli spawn, ε=0 greedy, successo = **500 step senza collisione**, su M1/M2/M3, `run_meta.json`, `aggregate_seeds.py`. L'agente Feng è allenato solo su M2 → **M1 e M3 diventano held-out** (generalizzazione zero-shot).

## 5. Come lanciare il training

```bash
# multi-seed, ognuno il suo seed; --config feng → artefatti in runs/feng/seed_<S>/
./start_train_feng.sh --seed=0 --config=feng
./start_train_feng.sh --seed=0 --config=feng --reset   # backup pre-reset automatico, poi pulisce
```
- M2-only, 3000 ep = 15 blocchi × 200, Gazebo riavvia ogni blocco (headless, gui:=false).
- **Backup prima di ogni reset**: lo script copia `runs/feng/seed_<S>/` in `ANALISI_TRAINING/<data>/pre_reset_feng_seed_<S>/` PRIMA di cancellare.
- Eval: `./start_test.sh --seed=<S> --config=feng --reps=30`.
- Aggregazione multi-seed: `python3 src/my_usv/scripts/aggregate_seeds.py --config feng --output ANALISI_TRAINING/<data>/aggregate_feng.csv` → mean±std / IQM / 95% CI. **Mai il max, mai single-run.**

### Protocollo campagna (come r_alpha)
5 seed (0-4) per macchina, distribuzioni **within-machine** separate, **mai poolare** seed cross-machine. Confronto Feng-vs-r_alpha sotto stesso protocollo.

## 6. ⚠️ Watch-item PRIMA della campagna completa: stabilità senza grad-clip

Aver rimosso il grad-clip (per fedeltà a Feng) + MSE + spike −1000 = **rischio di divergenza dei Q** (dichiarato come Rischio #1 nello spec §5.3). Smoke Gazebo-free già fatto (500 `learn()` sintetici: loss finita, |Q|=1.8 → ok), **ma non conclusivo**.

**Prima di impegnare 4 PC × multi-seed × ore**, lanciare **UN blocco smoke** su una macchina e controllare `avg_loss` in `runs/feng/seed_0/training_log.csv` / `logs/feng_block_01_maze_2.log`:
- loss **finita** e non monotòna crescente oltre ~1e4 → OK, via libera alla campagna;
- loss **NaN/esplode** → STOP, reintrodurre `clip_grad_norm_(self.q_net.parameters(), 10.0)` in `train_core.py` (`learn()`), ricommittare, **rifare il freeze** e ripartire.

Non saltare questo passo: una divergenza scoperta dopo aver lanciato tutte le macchine = ore sprecate.

## 7. Stato test
Tutti i test passano **in isolamento**: `test_usv_logic` (14), `test_agent` (11), `test_usv_env` (12). La suite *full* mostra un fallimento di `test_agent` dovuto a un artefatto torch (`wait_tensor already registered`) a tempo di collection — **pre-esistente**, problema di isolamento pytest, non causato da questo branch (verificabile: `test_agent.py` da solo passa).

## 8. Commit dell'implementazione (branch `paper_metric_feng`)
`4e03c2f` STATE_DIM 50 · `4ac9f66` process_lidar uniforme · `5130ddf` reward puro · `9f72b8a` stato 50/no frame-stack/heading/DR/norm · `192e6db` no grad-clip · `c36cfa0` `start_train_feng.sh` · `d003e64`+`6d6d81d`+`4493e5f` fix test/commenti (review).
Review indipendente: **READY** (0 Critical; 1 Important fixato). Spec coverage 8/8, pipeline coerente, eval/train.py provati invariati.

## 9. FREEZE
Questo branch è **congelato** al commit di questo briefing (git_sha riportato al push). Utente + colleghi trainano la baseline Feng **a questo git_sha** → seed confrontabili. Nessun commit al codice di training/eval finché la campagna multi-seed Feng non è raccolta. Fix urgenti → branch separato.

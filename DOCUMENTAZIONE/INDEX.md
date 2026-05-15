# USV DDQN — Documentazione di progetto

**Stato corrente (2026-05-11):** branch `merge11_05` implementato — multi-maze interleaved training (M1+M2, 5000 ep, random spawn). Codice pronto, training non ancora avviato.  
Prossimo step operativo: `./test_spawns.sh 1` (valida spawn M1) → `./start_train_multimaze.sh --reset`.

---

## Leggi in questo ordine

| File | Contenuto | Per chi |
|------|-----------|---------|
| [ARCHITETTURA.md](ARCHITETTURA.md) | Cosa fa il sistema, come è fatto | Chiunque, primo da leggere |
| [ESPERIMENTI.md](ESPERIMENTI.md) | Tutti i training fatti, risultati, confronto | Capire la storia del progetto |
| [DECISIONI.md](DECISIONI.md) | Perché abbiamo scelto X invece di Y | Non rifare errori già discussi |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Cosa fare nella prossima iterazione | Chi riprende il lavoro |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | Problemi noti e soluzioni | Quando qualcosa non funziona |

---

## Guide operative

| File | Branch | Contenuto |
|------|--------|-----------|
| [GUIDA_OPERATIVA.md](GUIDA_OPERATIVA.md) | `feng_direct` | Come avviare training e test (principale) |
| [START_HERE.md](START_HERE.md) | tutti | Setup Docker/WSL2/X11 da zero |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | `corrections_claude` | Guida training vecchio branch (storica) |

---

## Report e analisi esperimenti

| File | Contenuto |
|------|-----------|
| [report_feng_direct.md](report_feng_direct.md) | Analisi dettagliata training `feng_direct` — cause di fallimento + letteratura |
| [risultati/](risultati/) | Log sessioni di analisi precedenti |

---

## Analisi paper e letteratura

Cartella: [`PAPER_ANALYSIS/`](PAPER_ANALYSIS/)

| File | Contenuto |
|------|-----------|
| [ANALISI_FIXED_FENG_FALLIMENTO.md](PAPER_ANALYSIS/ANALISI_FIXED_FENG_FALLIMENTO.md) | Diagnosi fallimento `fixed_feng`: perché Huber+clip+PER peggiorano, errori in ANALISI_PARAMETRI_FENG.md, gap con risultati Feng 2021, 9 reference bibliografiche |
| [report_spawn_generalizzazione_DRL.md](PAPER_ANALYSIS/report_spawn_generalizzazione_DRL.md) | Review letteratura su spawn diversity e generalizzazione |
| [[old] ANALISI_PARAMETRI_FENG.md](PAPER_ANALYSIS/[old]%20ANALISI_PARAMETRI_FENG.md) | Analisi parametri Feng 2021 (Matteo Bolo) — contiene errori, superata |
| [P5_Feng2021_CollisionAvoidance.pdf](PAPER_ANALYSIS/P5_Feng2021_CollisionAvoidance.pdf) | Paper Feng et al. 2021 |

---

## File storici

| File | Nota |
|------|------|
| [PrimoTraining.md](PrimoTraining.md) | Osservazioni primo training (pre-refactor) |
| [modifiche_claude.md](modifiche_claude.md) | Changelog branch `prova_claude_code` |
| [labirinti_coordinate.md](labirinti_coordinate.md) | Coordinate spawn originali per world launch |

---

## Stato branch

| Branch | Stato | Note |
|--------|-------|------|
| `main` | Stabile | Baseline, infrastruttura Docker/ROS2 |
| `paper_implementation` | Storico — FALLITO | Training 6115 ep, crash >85% in test |
| `feng_direct` | Completo — analizzato | Training 3000 ep M2 only, avg100=+391, 3/30 successi M2 |
| `fixed_feng` | Analizzato — FALLITO | Modifiche Huber+clip+batch256 errate — avg100 < 0 dopo 3000 ep |
| `merge11_05` | **Attivo — da avviare** | Multi-maze interleaved, 5000 ep, random spawn M1+M2. Codice pronto. |

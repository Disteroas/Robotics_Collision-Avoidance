# USV DDQN — Documentazione di progetto

**Stato corrente (2026-05-08):** training `feng_direct` completato, risultati analizzati.  
Crash rate test: Maze 2 = 90%, Maze 1/3 = 100%. Cause identificate. Prossimi step definiti.

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

## Report e analisi

| File | Contenuto |
|------|-----------|
| [report_feng_direct.md](report_feng_direct.md) | Analisi dettagliata training `feng_direct` — cause di fallimento + letteratura |
| [report_spawn_generalizzazione_DRL.md](report_spawn_generalizzazione_DRL.md) | Review letteratura su spawn diversity e generalizzazione |
| [risultati/](risultati/) | Log sessioni di analisi precedenti |

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
| `feng_direct` | **Attivo** | Training 3000 ep completato, analizzato |

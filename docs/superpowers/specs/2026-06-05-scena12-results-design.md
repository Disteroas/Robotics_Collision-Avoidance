# Scena 12 — Results: Feng vs r_alpha (design)

**Data:** 2026-06-05
**Segmento:** 12 — `[7:00 → 7:53]`, durata **53s**
**Lingua:** voice-over EN, testi a schermo EN, note IT
**Stile:** template scene 9-10 (`video/slide.py`): bg bianco + griglia quadretti ~46px (#f1f2f5) + glow blu angoli, rule full-width con segmento blu sx, Arial, eyebrow small-caps + headline INK (#16263d) + clausola blu, card bianche con barra blu `#2f6df0`. Palette dati **Feng rosso `#d62728` / r_alpha blu `#1f77b4`**. Niente logo Polito / niente emoji nei plot.

## Obiettivo

Raccontare in 53s tutti i risultati Feng vs r_alpha, **fedeli al report** (`latex/main.tex`). Quattro temi, riordinati per **seguire l'audio** (non l'ordine 1-2-3-4 del brief):
metrica 500-step → maze di training → bimodalità M3 → hardware → cause schianti.

## Fedeltà al report (verificato)

| Affermazione | Fonte report | Stato |
|---|---|---|
| M2: Feng 32.5 → r_alpha 42.6 (mean±std 19.5 / 14.0) | `main.tex:262` | ✅ |
| M3: Feng 29.0 → r_alpha 59.3 (std 46.8 / 43.3) | `main.tex:265` | ✅ |
| M3 bimodale entrambi → si **contano** i seed, no media/IQM | `main.tex:239-251` | ✅ |
| 3/10 → 7/10 seed generalizzano | `main.tex:45` | ✅ |
| HW-B → M3 collassa ~0%, M2 stabile | `main.tex:317-323` | ✅ |
| Schianti 2 classi: **Percettivo** (~¾) / **Cinematico** (~¼) | `main.tex:306-310, 327` | ✅ |
| Pocket cinematico = spawn M2 `(-7.0,5.0)`, R_min=0.625m, unwinnable, crasha 30/30 su tutti i 10 seed | `main.tex:324-329` + dati `eval_crashes_m2` | ✅ |

**⚠️ SCARTATO (non nel report, non inventare):** la tassonomia `Side 43 / Frontal 25 / Kinematic 19 / Dead-end 13` della vecchia scaletta. Il report ha solo lo split a 2 classi. Box/violin **solo su M2** (graduale); su M3 bimodale il report dice esplicito che l'IQM "describes no actual run" → M3 va a dot plot + conteggio.

## Il montaggio — 5 beat, canvas unico template

| # | Beat (VO) | ~s | Visual | Testo a schermo | Dato |
|---|---|---|---|---|---|
| A | Metrica 500-step | ~13 | Tabella risultati che nasce + mini-mappa M1/M2 ghost di copertura. **Nessuna card dedicata.** | eyebrow `Objective: survive 500 steps = full training-length coverage` | — |
| B | Training maze, fair fight | ~8 | Riga **M2: 32.5 → 42.6** + **violin/box con IQM (solo M2)** | caption `A small, honest win.` | `eval_summary` N=10, M2 |
| C | **M3 bimodale (centro scena)** | ~14 | **Dot plot 10 seed/config**: r_alpha 7 alti / 3 a 0, Feng 3 / 7. Niente media. | overlay `We don't average. We count → 7/10 vs 3/10` · caption `Still a lottery.` | `eval_summary` N=10, M3 per-seed |
| D | Hardware | ~7 | Due **finestre incorniciate** (placeholder Gazebo): HW-A vs HW-B → **M3: 59% vs 0%** | rosso `Same code. Different PC.` | report cross-machine |
| E | Schianti | ~11 | Due **finestre affiancate** (placeholder Gazebo M2) + **striscia azioni reale sotto ENTRAMBE**. Sx `Perceptual`, dx `Kinematic`. | caption `The problem was never the driving. It was the seeing.` | `eval_steps_m2` (2 episodi) |

VO verbatim = blocco AUDIO di `MATERIALE_VIDEO/scaletta.md` SEGMENTO 12 (non modificato).

## Punto 4 — clip schianti (decisione utente: opzione 1, entrambe su M2)

Le finestre robot sono **placeholder** (l'utente monta il footage Gazebo dopo in DaVinci). La **striscia azioni sotto è dato reale** da `eval_steps_m2.csv`, scelta da me:

- **Percettivo:** episodio M2 in cui il robot sterza con azione ~costante dentro un muro pur con spazio libero adiacente (es. `last_actions` tipo `4,4,4,4,4` o `0,0,0,0,0`, `crash_sector` left/right/front). Seed/episodio esatto pinnato in build scansionando `eval_crashes_m2`.
- **Cinematico:** spawn M2 **`(-7.0,5.0)`** (pocket R_min, crasha 30/30 su tutti i seed). Robot entra nel vicolo, non riesce a girare, sbatte.

**Striscia azioni:** barra delle 11 azioni discrete (-0.8…+0.8 rad/s), l'azione scelta evidenziata che avanza nel tempo, sincronizzata al replay. Stile template (blu, no emoji). Una sotto ogni finestra.

Per il match perfetto l'utente **ri-cattura quei 2 episodi** col GUI eval (deterministico → stesso episodio) e li dropa nelle finestre. Comandi esatti (Git Bash, root progetto, XLaunch attivo):

```bash
# Cinematico — spawn pocket (-7.0,5.0), M2, modello feng seed da pinnare (S_KIN)
./start_test_gui.sh 2 <S_KIN> feng_hw_A 1 1
# Percettivo — M2, modello+seed da pinnare (S_PERC)
./start_test_gui.sh 2 <S_PERC> <CONFIG_PERC> 1 1
# (firma: start_test_gui.sh [maze_id] [seed] [config] [reps] [speed]; 1x = real-time)
# Output GUI in runs/<config>/seed_<S>/gui/ (NON tocca eval_summary reale)
```

`<S_KIN>`, `<S_PERC>`, `<CONFIG_PERC>` vengono fissati in build dopo lo scan; round-robin deterministico → l'episodio mostrato è riproducibile a ogni run. OBS registra la GUI Gazebo a 1x.

## Produzione (pattern solito, host-side Python 3.14)

File nuovi in `DOCUMENTAZIONE/report_feng_vs_ralpha/video/`, output in `MATERIALE_VIDEO/scena_12/`:

| File | Beat | Contenuto |
|---|---|---|
| `anim_results_table.py` | A+B | tabella M2 (32.5→42.6) + eyebrow metrica + mini-mappa ghost + violin/box IQM M2 |
| `anim_bimodal_m3.py` | C | dot plot 10 seed/config M3, overlay conteggio 7/10 vs 3/10 |
| `anim_hw_collapse.py` | D | due finestre incorniciate, M3 59% vs 0%, rosso "Same code. Different PC." |
| `anim_crash_modes.py` | E | due finestre placeholder + 2 strisce azioni reali (percettivo/cinematico) |
| `build_scene12_montage.py` (opz.) | tutti | cuce i beat a ritmo VO, xfade, progress-bar opzionale come scena 9-10 |

**Riuso:** loader dati da `make_campaign_figures.py` (`load_crash_sectors`, lettura `eval_summary`/`eval_steps`), helper `slide.py` (`bg/headline/eyebrow/accent_card/bottom_rule`), `vfx.py` (ramp/ease/lerp). Vincolo yuv420p → figsize×dpi pari (`style.writer` ha già `scale=trunc(iw/2)*2`).

**Dati N=10:** `runs/feng_hw_A/seed_0..9/` e `runs/r_alpha_hw_A/seed_0..9/` (seed 5-9 r_alpha originariamente in repo `ralpha_extend_A`, già mergiati). Aggregati in `ANALISI_TRAINING/`.

## Ordine & sync audio (note montaggio)

- Visual segue l'audio: A→B→C→D→E.
- Centro scena = **C (bimodalità M3)**: massimo tempo/peso visivo.
- Color grade: caldo pieno/maturo (coerente con scene 9-13).
- Le finestre placeholder D ed E hanno la stessa cornice usata a fine scena 10 (stile coerente per il dropping Gazebo in DaVinci).

## Deferred / da risolvere in build

- Pin esatto `<S_PERC>`/`<CONFIG_PERC>` per il crash percettivo M2 (scan `eval_crashes_m2`, scegliere episodio leggibile).
- Pin `<S_KIN>` per il pocket `(-7.0,5.0)` (qualunque seed feng va: crash deterministico).
- Conferma valori per-seed M3 (7 vs 3 r_alpha; 3 vs 7 feng) dal `eval_summary` aggregato prima di hardcodare il dot plot.
- `build_scene12_montage.py` con/senza progress-bar = file separati (come scena 9-10).

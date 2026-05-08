"""
Analisi training paper_implementation — 2026-05-08

Struttura training (3 blocchi distinti nel CSV):
  Block 0 (rows 0-2999):   3000 ep, BETA_DECAY=0.995 (OLD — codice non aggiornato all'avvio)
  Block 1 (rows 3000-3114): 115 ep, BETA_DECAY=0.999 (NEW, interrotto)
  Block 2 (rows 3115-6114): 3000 ep, BETA_DECAY=0.999 (NEW, completo)

Conclusioni principali: vedi sezione CONCLUSIONI in fondo.
"""

import csv
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.dirname(SCRIPT_DIR)
TRAIN_CSV    = os.path.join(REPO_ROOT, "src", "my_usv", "scripts", "training_log.csv")
TEST_CSV     = os.path.join(REPO_ROOT, "src", "my_usv", "scripts", "test_results.csv")
OUT_DIR      = os.path.join(SCRIPT_DIR, "plots_paper_impl")
os.makedirs(OUT_DIR, exist_ok=True)

# ── load ─────────────────────────────────────────────────────────────────────
with open(TRAIN_CSV) as f:
    all_rows = list(csv.DictReader(f))

with open(TEST_CSV) as f:
    test_rows = list(csv.DictReader(f))

# ── block segmentation ───────────────────────────────────────────────────────
# ep_global resets to 1 at each block boundary
blocks = []
prev_ep = 0
seg_start = 0
for i, r in enumerate(all_rows):
    ep = int(r["ep_global"])
    if ep < prev_ep:
        blocks.append(all_rows[seg_start:i])
        seg_start = i
    prev_ep = ep
blocks.append(all_rows[seg_start:])  # last block

assert len(blocks) == 3, f"Expected 3 blocks, got {len(blocks)}"
B0, B1, B2 = blocks  # Block0=old, Block1=short, Block2=new

# ── helpers ──────────────────────────────────────────────────────────────────
def crash_rate_windows(rows, window=100):
    """Rolling crash rate (window non-overlapping)."""
    rates, ep_mids = [], []
    for i in range(0, len(rows), window):
        w = rows[i:i+window]
        if not w:
            continue
        rates.append(sum(1 for r in w if r["crashed"] == "1") / len(w) * 100)
        ep_mids.append(int(w[len(w)//2]["ep_global"]))
    return ep_mids, rates


def rolling_mean(vals, n=50):
    out = []
    for i in range(len(vals)):
        lo = max(0, i - n + 1)
        out.append(np.mean(vals[lo:i+1]))
    return out


def eps_theoretical(n_eps, beta, eps0=1.0, eps_min=0.05):
    return [max(eps0 * beta**i, eps_min) for i in range(n_eps)]


# ── figure colours ───────────────────────────────────────────────────────────
C_M1   = "#2196F3"   # blue  — maze 1
C_M2   = "#F44336"   # red   — maze 2
C_M3   = "#4CAF50"   # green — maze 3
C_B0   = "#9E9E9E"   # grey  — block 0 old
C_B2   = "#FF9800"   # orange — block 2 new


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — epsilon comparison: old vs new, actual vs theoretical
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 1 — Epsilon decay: Block 0 (old) vs Block 2 (new)", fontsize=13)

for ax, block, label, color, beta in [
    (axes[0], B0, "Block 0 — BETA_DECAY=0.995 (OLD)", C_B0, 0.995),
    (axes[1], B2, "Block 2 — BETA_DECAY=0.999 (NEW)", C_B2, 0.999),
]:
    eps_actual = [float(r["epsilon"]) for r in block]
    ep_nums    = [int(r["ep_global"]) for r in block]
    eps_theory = eps_theoretical(max(ep_nums), beta)

    ax.plot(ep_nums, eps_actual, color=color, lw=1.2, label="ε effettivo (log)")
    ax.plot(range(1, max(ep_nums)+1), eps_theory[:max(ep_nums)],
            color="black", ls="--", lw=1, label=f"ε teorico β={beta}")

    # mark phase 2 trigger
    m2_start = next((r for r in block if r["maze"] == "2"), None)
    if m2_start:
        ep_p2 = int(m2_start["ep_global"])
        eps_p2 = float(m2_start["epsilon"])
        ax.axvline(ep_p2, color="purple", ls=":", lw=1.5)
        ax.annotate(f"Phase 2\nep={ep_p2}\nε={eps_p2:.3f}",
                    xy=(ep_p2, eps_p2), xytext=(ep_p2 + 80, eps_p2 + 0.12),
                    fontsize=8, color="purple",
                    arrowprops=dict(arrowstyle="->", color="purple"))

    ax.set_title(label, fontsize=10)
    ax.set_xlabel("Episode (ep_global)")
    ax.set_ylabel("ε")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig1_epsilon.png"), dpi=120)
plt.close()
print("Saved fig1_epsilon.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — crash rate: B0 vs B2, per maze
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Figura 2 — Crash rate (finestre da 100 ep) — Block 0 vs Block 2", fontsize=13)

for col, (block, blabel) in enumerate([(B0, "Block 0 (OLD β=0.995)"),
                                        (B2, "Block 2 (NEW β=0.999)")]):
    m1 = [r for r in block if r["maze"] == "1"]
    m2 = [r for r in block if r["maze"] == "2"]

    # maze 1
    x1, y1 = crash_rate_windows(m1, 100)
    axes[0][col].bar(range(len(y1)), y1, color=C_M1, alpha=0.7, width=0.8)
    axes[0][col].axhline(np.mean(y1), color=C_M1, ls="--", lw=1.5,
                         label=f"media {np.mean(y1):.1f}%")
    axes[0][col].set_title(f"{blabel} — Maze 1 crash rate", fontsize=9)
    axes[0][col].set_xlabel("Finestra 100-ep")
    axes[0][col].set_ylabel("Crash %")
    axes[0][col].set_ylim(0, 105)
    axes[0][col].legend(fontsize=8)
    axes[0][col].grid(alpha=0.3)

    # maze 2
    x2, y2 = crash_rate_windows(m2, 100)
    axes[1][col].bar(range(len(y2)), y2, color=C_M2, alpha=0.7, width=0.8)
    if y2:
        axes[1][col].axhline(np.mean(y2), color=C_M2, ls="--", lw=1.5,
                             label=f"media {np.mean(y2):.1f}%")
    axes[1][col].set_title(f"{blabel} — Maze 2 crash rate", fontsize=9)
    axes[1][col].set_xlabel("Finestra 100-ep")
    axes[1][col].set_ylabel("Crash %")
    axes[1][col].set_ylim(0, 105)
    axes[1][col].legend(fontsize=8)
    axes[1][col].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig2_crash_rate.png"), dpi=120)
plt.close()
print("Saved fig2_crash_rate.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — reward rolling mean in Block 2, split by maze
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 3 — Reward (media mobile 50 ep) — Block 2 (NEW code)", fontsize=13)

m1_b2 = [r for r in B2 if r["maze"] == "1"]
m2_b2 = [r for r in B2 if r["maze"] == "2"]

for ax, rows_m, color, maze_label in [
    (axes[0], m1_b2, C_M1, "Maze 1"),
    (axes[1], m2_b2, C_M2, "Maze 2"),
]:
    rewards = [float(r["reward"]) for r in rows_m]
    ep_nums = list(range(1, len(rewards) + 1))
    rm = rolling_mean(rewards, 50)

    ax.plot(ep_nums, rewards, color=color, alpha=0.25, lw=0.7, label="reward")
    ax.plot(ep_nums, rm, color=color, lw=2, label="media mobile 50")
    ax.axhline(0, color="black", ls="--", lw=0.8)
    ax.axhline(np.mean(rewards), color=color, ls=":", lw=1.2,
               label=f"media totale {np.mean(rewards):.0f}")

    # Phase 2 threshold reference
    if maze_label == "Maze 1":
        ax.axhline(1500, color="purple", ls="-.", lw=1, label="PHASE2_THRESHOLD=1500")

    ax.set_title(f"Block 2 — {maze_label} reward", fontsize=10)
    ax.set_xlabel(f"Episodio ({maze_label})")
    ax.set_ylabel("Reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig3_reward_b2.png"), dpi=120)
plt.close()
print("Saved fig3_reward_b2.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — loss trend Block 2 + avg steps per ep
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 4 — Loss e steps medi (Block 2)", fontsize=13)

ep_b2      = [int(r["ep_global"]) for r in B2]
losses_b2  = [float(r["avg_loss"]) for r in B2]
steps_b2   = [int(r["steps"]) for r in B2]

# Phase 2 start in B2
ep_phase2_b2 = int(next(r for r in B2 if r["maze"] == "2")["ep_global"])

for ax, vals, label, color in [
    (axes[0], losses_b2, "avg_loss", "#9C27B0"),
    (axes[1], steps_b2,  "steps/ep", "#FF9800"),
]:
    rm = rolling_mean(vals, 50)
    ax.plot(ep_b2, vals, color=color, alpha=0.2, lw=0.5)
    ax.plot(ep_b2, rm,   color=color, lw=2, label=f"{label} (media 50)")
    ax.axvline(ep_phase2_b2, color="purple", ls=":", lw=1.5,
               label=f"Phase 2 ep={ep_phase2_b2}")
    ax.set_xlabel("ep_global")
    ax.set_ylabel(label)
    ax.set_title(f"Block 2 — {label}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig4_loss_steps.png"), dpi=120)
plt.close()
print("Saved fig4_loss_steps.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — test results
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle("Figura 5 — Risultati test (30 ep × 3 maze)", fontsize=13)

for col, maze_id in enumerate(["1", "2", "3"]):
    m_rows = [r for r in test_rows if r["maze_id"] == maze_id]
    steps  = [int(r["steps"]) for r in m_rows]
    colors = [C_M1 if r["crashed"] == "0" else "#EF9A9A" for r in m_rows]
    success_n = sum(1 for r in m_rows if r["crashed"] == "0")
    crash_n   = len(m_rows) - success_n

    axes[col].bar(range(1, len(steps) + 1), steps, color=colors, edgecolor="none")
    axes[col].axhline(500, color="black", ls="--", lw=1, label="MAX_STEPS=500")
    axes[col].set_title(
        f"Maze {maze_id}: {success_n}/30 successi ({100*success_n/30:.0f}%)\n"
        f"crash: {crash_n}/30 ({100*crash_n/30:.0f}%)",
        fontsize=10,
    )
    axes[col].set_xlabel("Test episode")
    axes[col].set_ylabel("Steps")
    axes[col].set_ylim(0, 550)
    axes[col].legend(fontsize=8)

    succ_patch  = mpatches.Patch(color=C_M1,     label=f"successo ({success_n})")
    crash_patch = mpatches.Patch(color="#EF9A9A", label=f"crash ({crash_n})")
    axes[col].legend(handles=[succ_patch, crash_patch], fontsize=8)
    axes[col].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig5_test_results.png"), dpi=120)
plt.close()
print("Saved fig5_test_results.png")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Block 2: reward early learning (pre-phase2)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle("Figura 6 — Block 2: Phase 1 (eps 1-400, solo Maze 1) — reward e crash", fontsize=12)

pure_m1_b2 = [r for r in B2 if r["maze"] == "1" and int(r["ep_global"]) <= 400]
ep_pure    = [int(r["ep_global"]) for r in pure_m1_b2]
rew_pure   = [float(r["reward"]) for r in pure_m1_b2]
crash_pure = [r["crashed"] == "1" for r in pure_m1_b2]

ax.scatter(ep_pure, rew_pure,
           c=[C_M2 if c else C_M1 for c in crash_pure],
           s=12, alpha=0.6, zorder=3)
ax.plot(ep_pure, rolling_mean(rew_pure, 20),
        color="black", lw=2, label="media mobile 20")
ax.axhline(1500, color="purple", ls="-.", lw=1.5, label="PHASE2_THRESHOLD=1500")
ax.axvline(400, color="orange", ls=":", lw=1.5, label="Phase 2 scritta (~ep250, Gazebo lag→ep401)")

succ_p = mpatches.Patch(color=C_M1, label="successo")
crash_p = mpatches.Patch(color=C_M2, label="crash")
handles = [succ_p, crash_p,
           plt.Line2D([0],[0], color="black", lw=2, label="media mobile 20"),
           plt.Line2D([0],[0], color="purple", ls="-.", lw=1.5, label="PHASE2_THRESHOLD")]
ax.legend(handles=handles, fontsize=8)
ax.set_xlabel("ep_global")
ax.set_ylabel("Reward")
ax.set_title(f"Phase 1 pura — crash rate: {sum(crash_pure)/len(crash_pure)*100:.1f}%", fontsize=10)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig6_phase1_detail.png"), dpi=120)
plt.close()
print("Saved fig6_phase1_detail.png")


# ════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY STATS
# ════════════════════════════════════════════════════════════════════════════
def crash_rate(rows):
    return sum(1 for r in rows if r["crashed"] == "1") / len(rows) * 100 if rows else 0.0


print("\n" + "="*60)
print("SOMMARIO STATISTICHE")
print("="*60)
print(f"\nBlock 0 ({len(B0)} ep, OLD beta=0.995):")
m1b0 = [r for r in B0 if r["maze"]=="1"]
m2b0 = [r for r in B0 if r["maze"]=="2"]
p2b0 = next((r for r in B0 if r["maze"]=="2"), None)
print(f"  Maze1: {len(m1b0)} ep, crash {crash_rate(m1b0):.1f}%")
print(f"  Maze2: {len(m2b0)} ep, crash {crash_rate(m2b0):.1f}%")
if p2b0:
    print(f"  Phase2 trigger: ep={p2b0['ep_global']} eps={float(p2b0['epsilon']):.4f} -> eps reset NON efficace (<0.5)")

print(f"\nBlock 1 ({len(B1)} ep, NEW beta=0.999, interrotto):")
print(f"  Maze1: {len(B1)} ep, crash {crash_rate(B1):.1f}%")
print(f"  eps start={B1[0]['epsilon']}, end={B1[-1]['epsilon']}")

print(f"\nBlock 2 ({len(B2)} ep, NEW beta=0.999):")
m1b2 = [r for r in B2 if r["maze"]=="1"]
m2b2 = [r for r in B2 if r["maze"]=="2"]
p2b2 = next((r for r in B2 if r["maze"]=="2"), None)
pure_m1_pre = [r for r in m1b2 if int(r["ep_global"])<=400]
print(f"  Maze1 totale: {len(m1b2)} ep, crash {crash_rate(m1b2):.1f}%")
print(f"  Maze2 totale: {len(m2b2)} ep, crash {crash_rate(m2b2):.1f}%")
print(f"  Phase 1 pura (ep1-400): crash {crash_rate(pure_m1_pre):.1f}%")
if p2b2:
    eps_p2 = float(p2b2["epsilon"])
    print(f"  Phase2 trigger: ep~250 (Gazebo lag->log ep={p2b2['ep_global']}), eps={eps_p2:.4f}")
    print(f"  eps reset: max({eps_p2:.3f}, 0.5) = {max(eps_p2, 0.5):.3f} -> reset NON eseguito (eps gia' > 0.5)")

print(f"\nTest results (checkpoint finale):")
for m in ["1","2","3"]:
    mr = [r for r in test_rows if r["maze_id"]==m]
    s  = sum(1 for r in mr if r["crashed"]=="0")
    print(f"  Maze {m}: {s}/30 ({100*s/30:.0f}%)")

print("\n" + "="*60)
print("CONCLUSIONI E DIAGNOSI")
print("="*60)
conclusions = """
ROOT CAUSE 1 - Block 0 girato con BETA_DECAY=0.995 (vecchio codice)
  eps=0.0812 quando Phase 2 e' partita (ep 501). Esplorazione quasi nulla su Maze 2.
  Risultato: Maze 2 crash rate 95.8%. Identico al failure di curriculum_learning.
  Il training era partito PRIMA del fix del world plugin, probabilmente sul branch
  sbagliato o senza rebuild.

ROOT CAUSE 2 - Phase 2 trigger su reward assoluta, non su success rate
  PHASE2_THRESHOLD=1500 e' superato quando avg_reward>1500, ma l'agente puo' avere
  avg_reward>1500 anche con crash rate 80%+ semplicemente sopravvivendo ~454 steps.
  In Block 2: phase 2 scatta a ep~250 con crash rate 80% su Maze 1.
  L'agente NON ha realmente imparato Maze 1 -- ha solo imparato a sopravvivere a lungo
  casualmente prima di crashare. La soglia non distingue "navigazione corretta" da
  "sopravvivenza per inerzia con epsilon alto".

ROOT CAUSE 3 - Catastrophic forgetting dopo Phase 2
  In Block 2, dopo ep 250, l'agente alterna Maze1 e Maze2. La crash rate Maze1
  non migliora (85-92%) per tutte le 1600 episodi. Il mixing Maze2 (88% crash)
  contamina il replay buffer e impedisce ulteriore apprendimento su Maze1.

ROOT CAUSE 4 - Reward asimmetria con spawn random
  +5/step vs -1000/crash: con spawn random, ogni episodio parte da posizione diversa.
  Con epsilon alto (esplorazione), il robot esegue azioni casuali -> crash rapidi
  (avg ~250 steps). Il segnale negativo (-1000) domina: Q-values convergono verso
  "stai fermo / evita tutto" piuttosto che navigare verso il goal.
  In Block 0 (spawn fisso), l'agente poteva memorizzare un unico percorso --
  ma con spawn random serve vera generalizzazione, difficile con solo -1000/crash.

ROOT CAUSE 5 - eps reset a Phase 2 non ha avuto effetto reale
  In Block 2, eps al trigger = 0.670 > 0.5, quindi max(0.670, 0.5) = 0.670: nessun
  cambio. La condizione e' corretta ma il timing (fase 2 a ep 250 con beta=0.999)
  fa si' che eps sia ancora abbastanza alto -- il reset e' irrilevante.

RISULTATI TEST
  Maze 1: 26.7%, Maze 2: 26.7%, Maze 3: 40.0%
  Sorprendente che Maze 3 (mai visto) superi Maze 1 e 2.
  Ipotesi: struttura Maze 3 aperta (meno ostacoli critici) = piu' facile per
  comportamento di default dell'agente (avanzare dritto). Maze 1 e 2 hanno
  corridoi stretti che richiedono decisioni precise.

RACCOMANDAZIONI
  1. THRESHOLD basato su success rate, non reward assoluta:
     Passare a Phase 2 solo se success_rate_50ep >= 0.6 (60% successi su finestra 50).
  2. Reward shaping con distanza/goal:
     Aggiungere componente positiva proporzionale alla distanza percorsa o alla
     vicinanza al goal -- segnale piu' denso, meno sparse.
  3. Spawn curriculum per Maze 1:
     Iniziare con spawn fisso vicino al goal, poi gradualmente aumentare distanza.
     Combina la stabilita' di spawn fisso con la generalizzazione graduale.
  4. Separare Phase 1 e Phase 2 in run distinte:
     Trainare Maze1 fino a 60%+ success rate, salvare checkpoint, poi fine-tuning
     su Maze2 separatamente. Evita catastrophic forgetting.
  5. Ridurre BETA_DECAY durante Phase 2 Maze 1 (ritenzione):
     Quando si introduce Maze 2, abbassare learning rate su Maze 1 per rallentare
     il forgetting.
"""
print(conclusions)

print(f"\nPlot salvati in: {OUT_DIR}")

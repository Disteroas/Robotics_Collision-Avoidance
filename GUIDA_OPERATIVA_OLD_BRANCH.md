# 🤖 Guida Operativa USV - Branch `old`

Questa repository contiene l'implementazione del metodo di evitamento collisioni basato su **Deep Reinforcement Learning (DDQN)** per un robot USV (*Unmanned Surface Vehicle*).

---

## 📋 Specifiche Tecniche del Progetto

* **Algoritmo:** Double Deep Q-Network (DDQN) per ridurre il bias di sovrastima.
* **Input:** 50 raggi LIDAR campionati uniformemente con portata massima di 5.0 metri.
* **Azioni:** 11 opzioni di sterzo discrete con velocità lineare fissa.
* **Funzione di Reward:** Implementata la logica originale del paper (senza reward shaping graduale):
    * `+5.0`: Per ogni time-step concluso senza collisioni.
    * `-1000.0`: In caso di collisione con un ostacolo.

---

## 🔄 1. Sincronizzazione Git (Workflow Sicuro)

Segui questi passaggi all'inizio di ogni sessione per evitare conflitti nel team:

```bash
# Entra nella cartella del progetto
cd ~/usv_ws

# Assicurati di essere sul branch dedicato
git checkout old

# Scarica le ultime modifiche dei colleghi
git pull origin old
```

---

## 🚀 2. Avvio Simulazione (Procedura a 2 Terminali)

A causa delle dipendenze grafiche di **Gazebo** e della comunicazione **ROS2**, è necessario utilizzare due terminali separati.

### **Terminale 1: Ambiente 3D (Gazebo)**
Questo terminale avvia il container Docker e visualizza il mondo fisico.

```bash
cd ~/usv_ws

# Abilita l'accesso alla grafica X11 (necessario una volta per sessione)
xhost +local:docker

# Avvia la simulazione (Scegli il labirinto: 1, 2 o 3)
./start_sim.sh 1
```
> **Nota:** Se i raggi laser non appaiono, attiva l'opzione dal menu di Gazebo: `View` -> `Laser Scans`.

### **Terminale 2: Intelligenza Artificiale (Training)**
Mentre Gazebo è attivo, apri un nuovo terminale per lanciare l'addestramento.

```bash
cd ~/usv_ws

# Avvia lo script di training Python dentro il container
./start_train.sh
```

---

## 💾 3. Salvataggio e Caricamento (Push)

A fine giornata, salva il tuo progresso su GitHub per non perdere il lavoro:

```bash
# Controlla i file modificati (es. usv_env.py)
git status

# Prepara i file per il commit
git add .

# Crea il pacchetto con un messaggio descrittivo
git commit -m "Ripristinato reward originale paper (+5/-1000) e script sim cross-platform"

# Carica online sul branch old
git push origin old
```

---

## 🛠️ 4. Risoluzione dei Problemi (Troubleshooting)

### **Errore Git:** `! [rejected] old -> old (fetch first)`
Questo errore si verifica quando qualcuno ha aggiunto nuovo codice al branch `old` su GitHub e la tua versione locale non è aggiornata. Git ti impedisce il push per evitare di sovrascrivere il lavoro altrui.

**Come risolvere (I due step):**

1. **Scarica e unisci le novità (Pull):**
   ```bash
   git pull origin old
   ```
   *Nota: Se si apre un editor di testo (Nano o Vim) per confermare un "Merge branch", salva e chiudi:*
   * **Nano:** Premi `Ctrl + O`, poi `Invio`, poi `Ctrl + X`.
   * **Vim:** Scrivi `:wq` e premi `Invio`.

2. **Riprova a inviare il tuo lavoro (Push):**
   ```bash
   git push origin old
   ```

---

## ⚙️ Note sul Sistema e Docker

* **Script `start_sim.sh`:** È stato reso "Cross-Platform". Riconosce automaticamente se stai lavorando su Linux (Ubuntu) o Windows e configura il driver grafico corretto (`DISPLAY`).
* **Volume Docker:** La cartella locale `~/usv_ws` è collegata direttamente a `/home/usv_ws` nel container. Ogni modifica ai file `.py` fatta sul tuo PC è immediatamente attiva dentro Docker.
* **Accelerazione Hardware:** Se la grafica di Gazebo risulta lenta su Ubuntu, assicurati che lo script utilizzi il flag `--device=/dev/dri` (per Intel/AMD) o `--gpus all` (per NVIDIA).# 🤖 Guida Operativa USV - Branch `old`


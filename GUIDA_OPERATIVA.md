# 🌊 Guida Operativa Quotidiana: Simulazione e Training USV

Questa guida spiega come avviare l'ambiente di lavoro ogni volta che riprendi il progetto. Grazie agli script di automazione, la procedura è rapida e riduce al minimo gli errori manuali.

---

## 🛠 1. Preparazione Ambiente (Windows)

Prima di aprire il terminale, assicurati che i "motori" siano accesi:

1.  **XLaunch (VcXsrv):** * Avvialo dal menu Start.
    * Seleziona **Multiple Windows** -> Avanti.
    * Seleziona **Start no client** -> Avanti.
    * **CRITICO:** Spunta la casella **"Disable access control"**. Senza questa, il robot non apparirà.
    * Clicca su Fine.
2.  **Docker Desktop:** Apri l'applicazione e attendi che l'icona della balena diventi **verde** ("Engine Running").

---

## 🚀 2. Fase 1: Avvio della Simulazione (Terminale 1)

Il primo terminale si occupa di generare l'universo fisico (Gazebo) e il labirinto.

1.  Apri **Git Bash** nella cartella del progetto (`Robotics_Collision-Avoidance`).
2.  Lancia lo script di avvio scegliendo il numero del labirinto che vuoi testare (1, 2 o 3):
    ```bash
    ./start_sim.sh 1
    ```
    *(Nota: Se non scrivi il numero, lo script caricherà il Labirinto 1 di default).*

**Cosa succede ora?**
Si aprirà la finestra di Gazebo su Windows. Vedrai l'acqua, il robot e le mura del labirinto scelto. Il terminale rimarrà "bloccato" a gestire la fisica del mondo. **Non chiuderlo.**

---

## 🧠 3. Fase 2: Avvio dell'Intelligenza Artificiale (Terminale 2)

Mentre Gazebo è aperto, dobbiamo dare vita al robot lanciando il pilota automatico (lo script di training).

1.  Apri una **SECONDA finestra di Git Bash** (sempre nella cartella del progetto).
2.  Lancia lo script per avviare l'addestramento:
    ```bash
    ./start_train.sh
    ```

**Cosa succede ora?**
Questo script entra automaticamente nel container già attivo e lancia il file `train.py`. Vedrai i log dell'intelligenza artificiale scorrere e, guardando la finestra di Gazebo, vedrai il robot iniziare a muoversi e imparare a schivare gli ostacoli.

---

## 🕹️ 4. Gestione dei Labirinti

Il tuo collega ha preparato 3 scenari. Puoi passare da uno all'altro semplicemente chiudendo la simulazione e riavviandola con il numero corrispondente:

* **Labirinto 1 (Base):** `./start_sim.sh 1` (Coordinate: x:-3, y:-5)
* **Labirinto 2 (Intermedio):** `./start_sim.sh 2` (Coordinate: x:-6, y:0)
* **Labirinto 3 (Avanzato):** `./start_sim.sh 3` (Coordinate: x:-2, y:-1)

*Nota: Lo script gestisce automaticamente il cambio dei percorsi dei file `.world` e le coordinate di partenza corrette.*

---

## 📝 5. Modifica del Codice (Workflow Consigliato)

Non è necessario chiudere tutto per modificare il codice:

1.  Apri i file Python (es. `train.py`) usando **VS Code** o un editor su Windows.
2.  Modifica il codice e **salva il file**.
3.  Le modifiche sono istantanee dentro Docker grazie al "ponte" (volume) che abbiamo creato.
4.  Per testare la modifica, vai nel **Terminale 2**, premi `Ctrl + C` per fermare lo script e rilancia `./start_train.sh`. Non serve riavviare Gazebo ogni volta!

---

## 🆘 6. Risoluzione Rapida Problemi

### Errore: "Conflict. The container name /usv_container is already in use"
Succede se la simulazione precedente è crashata o non è stata chiusa bene.
* **Soluzione:** Digita `docker rm -f usv_container` e riprova.

### Errore: "Display not found" o schermo nero in Gazebo
Quasi certamente hai dimenticato di spuntare "Disable access control" in XLaunch.
* **Soluzione:** Chiudi XLaunch dalla barra delle icone di Windows (vicino all'orologio) e riavvialo seguendo attentamente la Fase 1.

### Errore: "Package my_usv not found"
Succede se i file non sono stati compilati dopo una modifica strutturale.
* **Soluzione:** Nel Terminale 2, digita `colcon build`, attendi la fine, poi digita `source install/setup.bash`.

---

## 🛑 7. Chiusura Sicura

Quando hai finito:
1.  Vai nel **Terminale 2** e premi `Ctrl + C`, poi digita `exit`.
2.  Vai nel **Terminale 1** e premi `Ctrl + C`. Il container si autodistruggerà pulendo la memoria grazie al parametro `--rm`.
3.  Chiudi Docker Desktop e XLaunch.
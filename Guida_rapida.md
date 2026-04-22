# Guida Rapida: Avvio Quotidiano della Simulazione

Questa guida spiega i passaggi da seguire ogni volta che accendi il PC per lavorare al progetto, assumendo che tu abbia già completato l'installazione iniziale.

---

## Procedura d'Avvio (Sequenza Obbligatoria)

Per far funzionare la simulazione grafica, devi seguire esattamente questo ordine:

### 1. Avviare il Server Grafico (X11)
Il robot ha bisogno di un "monitor" dove mostrare la simulazione.
1. Cerca **XLaunch** nel menu Start e avvialo.
2. Segui i passaggi (o usa il file di configurazione salvato):
   - **Multiple Windows** -> Avanti.
   - **Start no client** -> Avanti.
   - **IMPORTANTE:** Spunta **"Disable access control"**. -> Avanti.
   - Clicca su **Fine**.
*L'icona della "X" nera deve essere visibile nella barra delle applicazioni (vicino all'orologio).*



### 2. Avviare Docker Desktop
1. Apri **Docker Desktop**.
2. Attendi che l'icona della balena in basso a sinistra diventi **verde** e compaia la scritta **"Engine Running"**.
*Non lanciare comandi finché Docker non è pronto.*

### 3. Aprire il Terminale (Git Bash)
1. Vai nella cartella del progetto sul tuo Desktop (`Robotics_Collision-Avoidance`).
2. Fai clic destro in uno spazio vuoto e seleziona **"Open Git Bash here"**.

---

## Esecuzione del Progetto

### Step A: Entrare nel Container (Il Ponte)
Incolla questo comando nel terminale Git Bash per attivare l'ambiente virtuale:

```bash
docker run -it \
  --env="DISPLAY=host.docker.internal:0.0" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/$(pwd):/home/usv_ws" \
  usv_rl_project
```

Se il prompt cambia in `root@xxxxxxxx:/home/usv_ws#`, sei dentro il cervello del robot.

### Step B: Caricare l'Ambiente ROS 2 (Source)
Prima di lanciare qualsiasi comando ROS 2, devi dire al sistema dove si trovano i pacchetti. Digita:

```bash
source /opt/ros/humble/setup.bash
```
*(Nota: Se avete già compilato il progetto, potrebbe servire anche `source install/setup.bash`)*.



### Step C: Lanciare la Simulazione
Per far apparire il robot nella mappa, usa il comando `ros2 launch` specificando il pacchetto e il file. Nel nostro caso:

```bash
ros2 launch my_usv spawn_robot.launch.py
```

---

## Come chiudere tutto correttamente

Quando hai finito di lavorare:

1. **Nel terminale:** Premi `Ctrl + C` per fermare la simulazione.
2. **Uscire dal container:** Digita `exit` e premi Invio.
3. **Spegnere Docker:** Fai clic destro sulla balena di Docker -> **Quit Docker Desktop**.
4. **Spegnere X11:** Fai clic destro sulla "X" nera -> **Exit**.

---

## Problemi veloci (Cheat Sheet)
- **Vedo errori "Display not found":** Non hai spuntato "Disable access control" in XLaunch. Chiudilo e riavvialo correttamente.
- **Il comando `ros2` non viene riconosciuto:** Hai dimenticato di digitare il comando `source` dello Step B.
- **Le modifiche al codice non si vedono:** Salva il file su Windows (es. su VS Code) e riavvia solo il comando di `launch` dentro il terminale. Grazie al `--volume`, i file sono sincronizzati in tempo reale.

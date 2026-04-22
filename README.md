# Guida Integrale al Setup dell'Ambiente di Simulazione USV

**Target:** Membri del Team Robotics  
**Scopo:** Configurazione da zero di Docker, WSL2 e X11 Forwarding su Windows per il progetto `Robotics_Collision-Avoidance`.

**Chat completa gemini**: [clicca qui](https://gemini.google.com/gem/4f2a3409d082/2447ed267e04600d)

---

## Indice
1. [Fase 1: GitHub e Preparazione File](#fase-1-github-e-preparazione-file)
2. [Fase 2: Hardware e Virtualizzazione (BIOS)](#fase-2-hardware-e-virtualizzazione-bios)
3. [Fase 3: Installazione di WSL2 e Linux](#fase-3-installazione-di-wsl2-e-linux)
4. [Fase 4: Configurazione Server Grafico (XLaunch)](#fase-4-configurazione-server-grafico-xlaunch)
5. [Fase 5: Docker Desktop - Installazione e Debug](#fase-5-docker-desktop---installazione-e-debug)
6. [Fase 6: Build e Lancio del Progetto](#fase-6-build-e-lancio-del-progetto)
7. [Fase 7: Risoluzione Errori Comuni (FAQ)](#fase-7-risoluzione-errori-comuni-faq)

---

## Fase 1: GitHub e Preparazione File

Prima di toccare il sistema, dobbiamo avere il codice e gli strumenti di base.

1. **Git:** Scarica e installa [Git for Windows](https://git-scm.com/). Durante l'installazione, lascia le opzioni predefinite (assicurati che "Git Bash" sia incluso).
2. **Account e Clone:**
   - Effettua il login su GitHub.
   - Crea una cartella di lavoro sul tuo PC (es: `C:\Progetto_Robotics`). **Evita percorsi con troppi spazi o caratteri speciali se possibile.**
   - Apri **Git Bash** in quella cartella (tasto destro in uno spazio vuoto -> *Open Git Bash here*).
   - Esegui il clone della repository:

```bash
git clone https://github.com/[URL_DELLA_REPOSTORY].git
```

---

## Fase 2: Hardware e Virtualizzazione (BIOS)

Docker non può girare se il processore non è autorizzato a creare macchine virtuali. 

1. **Controllo Preventivo:**
   - Premi `Ctrl + Shift + Esc` (Gestione Attività).
   - Vai su **Prestazioni** -> **CPU**.
   - Cerca la voce **Virtualizzazione** in basso a destra. 
   - **Se è "Disabilitata"**, devi entrare nel BIOS.
2. **Accesso al BIOS (Esempio per schede ASUS):**
   - Riavvia il PC.
   - Appena lo schermo si accende, premi ripetutamente `F2` o `Canc` finché non entri nel BIOS.
   - Premi `F7` per entrare in **Advanced Mode**.
   - Vai nella scheda **Advanced** -> **CPU Configuration**.
   - Cerca **Intel Virtualization Technology** (per CPU Intel) o **SVM Mode** (per CPU AMD).
   - Imposta il valore su **Enabled**.
   - Premi `F10` per confermare, salvare e uscire. Il PC si riavvierà.

---

## Fase 3: Installazione di WSL2 e Linux

Windows ha bisogno di un "cuore" Linux per far girare Docker correttamente.

1. Apri **PowerShell come Amministratore** (cerca "PowerShell" nel menu Start, fai clic destro -> *Esegui come amministratore*).
2. Digita il seguente comando per installare il sottosistema Linux e premi Invio:

```powershell
wsl --install -d Ubuntu
```

3. **Riavvia fisicamente il PC** (questo passaggio è obbligatorio per applicare le modifiche al kernel).
4. Al riavvio, si aprirà in automatico una finestra nera di Ubuntu. Inserisci uno username a tua scelta (es: `robot`) e una password semplice (es: `1234`). *Nota: mentre digiti la password non vedrai caratteri a schermo, è normale.*
5. Quando vedi il prompt verde (es: `robot@computer:~$`), l'installazione è finita. Digita `exit` per chiudere la finestra.

---

## Fase 4: Configurazione Server Grafico (XLaunch)

Il container Docker è "cieco". Per poter vedere la simulazione 3D (come Gazebo o RViz) sul monitor di Windows, ci serve un server X11 chiamato VcXsrv.

1. Scarica e installa **VcXsrv** da [SourceForge](https://sourceforge.net/projects/vcxsrv/).
2. Dal menu Start, cerca e avvia il programma **XLaunch**.
3. **Configurazione Step-by-Step:**
   - Schermata 1 (*Display settings*): Seleziona **Multiple Windows**, metti *Display Number* su **-1**. Clicca *Avanti*.
   - Schermata 2 (*Client startup*): Seleziona **Start no client**. Clicca *Avanti*.
   - Schermata 3 (*Extra settings*): **IMPORTANTE!** Oltre alle spunte già presenti, **DEVI obbligatoriamente spuntare "Disable access control"**. Se lo dimentichi, il container non potrà trasmettere il video a Windows e lo schermo resterà nero. Clicca *Avanti*.
   - Schermata 4: Clicca su **Save Configuration** (salva il file sul desktop per poter avviare questa configurazione con un doppio clic in futuro) e infine clicca su **Fine**.
4. Se ti appare una finestra minacciosa del **Firewall di Windows**, spunta **entrambe** le caselle (Reti Pubbliche e Private) e clicca su **Consenti accesso**.

---

## Fase 5: Docker Desktop - Installazione e Debug

1. Scarica [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Esegui l'installer come amministratore. Assicurati che la casella **"Use WSL 2 instead of Hyper-V"** sia **spuntata**. Lascia finire l'installazione e avvia Docker.
3. **Problema Noto: "Engine Starting" infinito**
   Se al primo avvio la balena in basso a destra (vicino all'orologio per intenderci) continua a girare per più di 3 minuti senza che la barra diventi verde:
   - Fai clic destro sull'icona della balena e seleziona **Quit Docker Desktop**.
   - Apri PowerShell (come Amministratore) e spegni il motore forzatamente digitando: `wsl --shutdown`.
   - Riavvia Docker Desktop. Questa volta dovrebbe agganciarsi in pochi secondi, mostrando "Engine running".

---

## Fase 6: Build e Lancio del Progetto

Ora colleghiamo tutto. **Usa sempre Git Bash per questi comandi**, per evitare gli errori di sintassi che causa PowerShell con i percorsi dei file.

1. **Spostati nella cartella del progetto:**
   Vai sul Desktop, entra nella cartella della repository clonata, fai clic destro in uno spazio vuoto e clicca su **Open Git Bash here**.

2. **Build dell'immagine:**
   Questo comando "cucina" il tuo ambiente leggendo le istruzioni nel file `Dockerfile` (scaricherà Ubuntu, ROS 2, e le librerie necessarie). Nel terminale digita:

```bash
docker build -t usv_rl_project .
```
*(Attendi pazientemente che arrivi al 100% e ti restituisca il cursore).*

3. **Lancio del Container (Il comando "ponte"):**
   Questo comando fa partire l'ambiente, abilita il flusso video verso XLaunch e collega la tua cartella di Windows a quella del robot in modo da sincronizzare il codice. Incolla questo:

```bash
docker run -it \
  --env="DISPLAY=host.docker.internal:0.0" \
  --env="QT_X11_NO_MITSHM=1" \
  --volume="/$(pwd):/home/usv_ws" \
  usv_rl_project
```

Se vedi che il terminale cambia e compare la scritta `root@xxxxxxxx:/home/usv_ws#`, complimenti: sei dentro il "cervello" del robot!

---

## Fase 7: Risoluzione Errori Comuni (FAQ)

### 1. Errore "500 Internal Server Error" o "Pipe not found"
Docker non riesce a comunicare con WSL perché il canale si è intasato.
- **Soluzione:** Chiudi Docker. Apri PowerShell e digita `taskkill /IM "Docker Desktop.exe" /F /T`, poi digita `wsl --shutdown`. Infine, riapri Docker Desktop.

### 2. Errore "HCS_E_HYPERV_NOT_INSTALLED" (in PowerShell)
Il BIOS ha la virtualizzazione disattivata oppure manca il componente "Piattaforma Macchina Virtuale" di Windows.
- **Soluzione:** Assicurati di aver fatto la Fase 2 nel BIOS. Poi apri PowerShell come Amministratore e digita `wsl.exe --install --no-distribution`. Riavvia il PC e ripeti la Fase 3.

### 3. Errore "bash: python: command not found" (dentro il container)
Nel container Linux moderno, il vecchio comando `python` è stato rimosso.
- **Soluzione:** Devi specificare la versione. Usa sempre `python3 nome_script.py`.

### 4. Errore "ValueError: Not a valid package name" (in ambiente ROS 2)
Stai cercando di lanciare un file di configurazione senza dire a ROS in quale pacchetto cercarlo.
- **Sintassi corretta in ROS 2:** `ros2 launch [nome_pacchetto] [nome_file.launch.py]`.
- **Esempio pratico:** `ros2 launch my_usv spawn_robot.launch.py`.

---
**Regola d'oro:** Ricordati di avviare **SEMPRE** la configurazione salvata di XLaunch prima di lanciare il comando `docker run`. Buon lavoro al team!

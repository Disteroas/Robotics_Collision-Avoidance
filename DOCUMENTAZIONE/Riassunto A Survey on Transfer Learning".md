# Riassunto: "A Survey on Transfer Learning" (Pan & Yang, 2010 - IEEE TKDE)

## 1. L'Obiettivo del Paper
Il Machine Learning tradizionale (e il Deep RL) si basa su un'assunzione matematica fortissima: **i dati di addestramento e i dati di test devono provenire dallo stesso "spazio delle feature" e avere la stessa "distribuzione"**. 
Se addestri un algoritmo a riconoscere le mele rosse, fallirà nel riconoscere le mele verdi. Se addestri un robot in un labirinto a griglia (Maze 2), fallirà in un labirinto curvo o aperto (Maze 1). 
Il **Transfer Learning (TL)** nasce per distruggere questo limite: è l'arte di estrarre la "conoscenza" appresa in un dominio sorgente (*Source Domain*) e riutilizzarla per risolvere un problema in un dominio di destinazione diverso ma correlato (*Target Domain*), evitando di dover riaddestrare la rete neurale da zero.

## 2. La Tassonomia: I 3 Tipi di Transfer Learning
Gli autori classificano il Transfer Learning in tre grandi famiglie:
1. **Inductive Transfer Learning:** Il dominio è lo stesso, ma il task cambia. (Es. Il robot sa già navigare nel labirinto, ora usa quella conoscenza per imparare a cercare una scatola rossa).
2. **Transductive Transfer Learning (o Domain Adaptation):** Il task è lo stesso, ma il dominio cambia. I dati non sono etichettati nel nuovo dominio. (Es. Il robot deve sempre evitare i muri, ma passa dal simulatore Gazebo al mondo reale fisico, oppure dal Maze 2 al Maze 3).
3. **Unsupervised Transfer Learning:** Nessuna etichetta disponibile né nel dominio sorgente né in quello target (spesso usato per clustering).

## 3. Cosa trasferire? (What to Transfer)
Una volta capito *quando* trasferire, il paper analizza *cosa* viene effettivamente passato da un'intelligenza all'altra:
* **Instance Transfer:** Riutilizzare pezzi di dati (esperienze) passati.
* **Feature-Representation Transfer:** Imparare a "guardare" i dati in modo universale.
* **Parameter Transfer:** Copiare i pesi (weights) della rete neurale del dominio sorgente e usarli come punto di partenza nel dominio target, assumendo che i modelli condividano dei parametri fondamentali (es. come processare i dati del LIDAR).
* **Relational-Knowledge Transfer:** Trasferire regole logiche.

---

# 🚀 LA SEZIONE PER NOI: Come usare questo paper nel tuo progetto

Questo paper vi fornisce il vocabolario accademico e la strategia per risolvere definitivamente il problema del Maze 1 e del Maze 3, oltre a darvi una base teorica per un eventuale passaggio su robot fisico (Sim-to-Real).

### 1. La diagnosi del fallimento (Il problema del "Covariate Shift")
Quando il vostro robot prende lo 0% di successi nel Maze 1 dopo essersi addestrato nel Maze 2, il prof potrebbe chiedervi: *"Perché? L'azione è sempre curvare, e il sensore è sempre il LIDAR"*.
* **Cosa direte voi (citando Pan & Yang):** *"Il robot subisce un fenomeno noto nel Transductive Transfer Learning come Covariate Shift (o Marginal Distribution Shift). Sebbene lo spazio delle feature (i 50 raggi del LIDAR) sia identico, la probabilità marginale $P(X)$ di vedere certe conformazioni geometriche nel Maze 1 è drasticamente diversa da quella del Maze 2. La rete neurale è andata in out-of-distribution."*

### 2. La Soluzione d'Oro: Parameter Transfer Learning (Il Fine-Tuning)
Invece di impazzire cercando di fare un addestramento mastodontico che includa contemporaneamente tutti e 3 i Maze (che su Gazebo rischia di essere lentissimo e instabile), potete usare il **Parameter Transfer**.
* **Come implementarlo in `train_core.py`:**
  1. Addestrate la vostra barca/cingolato sul **Maze 2** (il più difficile) fino a fargli raggiungere un ottimo livello (es. avg100 = 15.000). Questo è il vostro *Source Domain*.
  2. Salvate i pesi della rete (`q_net.pth`).
  3. Cambiate il mondo in Gazebo mettendo il **Maze 3** (il *Target Domain*).
  4. Invece di inizializzare la rete con pesi casuali, usate la funzione `load_ckpt` per caricare il cervello del Maze 2 nel robot che si trova nel Maze 3.
  5. Abbassate il Learning Rate (es. da `0.00025` a `0.00005`) e fatelo addestrare sul Maze 3.
* **Il Risultato:** Invece di metterci 6000 episodi per imparare il Maze 3, ci metterà magari solo 200 episodi. Avrete applicato con successo il *Parameter-based Transfer Learning*, dimostrando una maturità progettuale enorme.

### 3. Trasformare il "Difetto" in una "Feature" per la Tesi
Nel report di Matteo, lo zero percentuale sui Maze mai visti sembrava un errore del codice. Ora, con questo paper in mano, si trasforma nel capitolo finale della vostra relazione:
* Potete dedicare un paragrafo chiamato **"Domain Adaptation tramite Parameter Transfer"**.
* Spiegate che addestrare un DRL puro (DQN) su un singolo labirinto non porta a una *policy universale* (a causa del Covariate Shift descritto da Pan & Yang).
* Mostrate i grafici: un grafico che fa vedere come il training "da zero" sul Maze 3 sia lento, e un grafico che mostra come il training sul Maze 3 *partendo dai pesi del Maze 2* (Transfer Learning) converga quasi istantaneamente. 
* Avrete letteralmente realizzato in modo accademico il "training multi-stadio" che Feng sognava di fare nei suoi Future Works!

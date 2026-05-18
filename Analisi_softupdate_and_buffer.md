# Dinamiche di Apprendimento nel DDQN: Soft Update, Esperienza e Generalizzazione

Questa sezione analizza le criticità architetturali emerse durante l'addestramento dell'USV (Policy Degradation e Overfitting Geometrico) e spiega matematicamente le soluzioni adottate allo stato dell'arte per stabilizzare il Deep Reinforcement Learning in ambienti parzialmente osservabili.

---

## 1. Il Problema della Rete Target: "Hard Update" vs "Soft Update"

Nell'algoritmo Double DQN (DDQN), vengono utilizzate due reti neurali strutturalmente identiche per calcolare l'errore di approssimazione (Loss) ed evitare la sovrastima sistematica dei valori Q (*maximization bias*):
1. **Main Network ($\theta_{main}$):** Sceglie l'azione corrente e viene aggiornata tramite discesa del gradiente a ogni singolo step temporale.
2. **Target Network ($\theta_{target}$):** Rimane stabile per un certo intervallo e serve esclusivamente a calcolare il valore atteso futuro (il "traguardo" matematico della funzione obiettivo).

Il valore target per l'aggiornamento dei pesi è definito dall'equazione di Bellman ottimale:
$$Y_i = R_{i+1} + \gamma Q_{target}\left(S_{i+1}, \arg\max_{a} Q_{main}(S_{i+1}, a; \theta_{main}); \theta_{target}\right)$$

### L'Hard Update e lo "Shock" della Loss
Nella configurazione con **Hard Update**, la Target Network viene congelata e aggiornata in blocco copiando integralmente i pesi della Main Network solo ogni $N$ step (ad esempio, $N = 1000$):
$$\theta_{target} \leftarrow \theta_{main}$$

Questo approccio introduce una forte discontinuità nel processo di ottimizzazione. Per 999 step la Main Network insegue un traguardo $Y_i$ statico. Al millesimo step, il traguardo cambia improvvisamente in modo violento. Questo provoca un picco istantaneo nella funzione di Loss (MSE). 

I gradienti derivanti da questo shock possono destabilizzare i pesi sinaptici già ottimizzati, distruggendo le traiettorie complesse precedentemente apprese. Questo fenomeno spiega matematicamente la *Policy Degradation* riscontrata nelle run instabili, dove il reward medio mobile crolla repentinamente negli stadi avanzati dell'addestramento.

### La Soluzione: Il Soft Update (Polyak Averaging)
Per garantire la stabilità della convergenza, si applica il **Soft Update**. La Target Network non viene più congelata, ma si aggiorna a *ogni singolo step* assorbendo solo una frazione infinitesima ($\tau$) dei pesi della Main Network:
$$\theta_{target} \leftarrow \tau \theta_{main} + (1 - \tau) \theta_{target}$$

Impostando un valore conservativo come $\tau = 0.005$, il panorama della Loss si sposta in modo continuo e armonioso. La Target Network si muove come una media mobile esponenziale dei pesi della rete principale, eliminando i transitori distruttivi e proteggendo la policy dai collassi strutturali.

---

## 2. La Correlazione tra Meccanismi di Aggiornamento e Overfitting Geometrico

L'**Overfitting Geometrico** si manifesta quando la rete neurale smette di astrarre le feature logiche dei sensori (es. *"se il corridoio si stringe, riduci l'angolo di virata"*) e inizia a mappare puntualmente le distanze euclidee specifiche del singolo ambiente (es. *"se il raggio LIDAR a 45° misura esattamente 1.12 metri, esegui l'azione 4"*).

Esiste un legame diretto tra l'instabilità dell'Hard Update e l'insorgenza dell'overfitting geometrico:
1. Quando l'algoritmo subisce lo shock dell'Hard Update, la stabilità globale della policy si rompe e l'errore balza verso l'alto.
2. Per minimizzare rapidamente questo errore improvviso, l'ottimizzatore seleziona la via matematicamente più efficiente a breve termine: si adatta alla situazione statistica ampiamente dominante nel dataset.
3. Poiché l'USV passa la stragrande maggioranza del tempo a navigare lungo i rettilinei del labirinto, la rete ottimizza i pesi per risolvere in modo maniacale quel percorso specifico, distruggendo le zone di attivazione neuronale dedicate alle curve strette e insolite. 

Il passaggio al Soft Update, mantenendo bassa la varianza della Loss, permette alla rete di conservare e rifinire le regole generali di navigazione anche per gli stati meno frequenti, riducendo drasticamente l'overfitting spaziale.

---

## 3. Il Replay Buffer: Campionamento Uniforme vs. Prioritized Experience Replay (PER)

La selezione delle esperienze passate per l'addestramento gioca un ruolo cruciale nella capacità di generalizzazione dell'agente.

### Il Limite del Campionamento Uniforme
In un Replay Buffer standard (strutturato come una coda uniforme), ogni transizione memorizzata ha la stessa identica probabilità di essere estratta per il calcolo del gradiente:
$$P(i) = \frac{1}{N}$$

Nel contesto della navigazione robotica in un labirinto, questo approccio introduce un severo campionamento distorto (*data bias*). La guida rettilinea parallela alle pareti è un task frequente e a basso errore, mentre la negoziazione di una curva a gomito o la correzione da una posizione di spawn complessa sono eventi rari e critici. 

Pescando uniformemente dal buffer, i mini-batch saranno quasi interamente composti da transizioni rettilinee. Di conseguenza, la rete neurale tenderà a ignorare le dinamiche delle curve, provocando fallimenti sistematici e precoci non appena l'USV viene testato in configurazioni di partenza non lineari.

### Il Paradigma del Prioritized Experience Replay (PER)
Il PER scardina il campionamento uniforme, legando la probabilità di estrazione di un ricordo al suo livello di "sorpresa" o difficoltà, misurato direttamente dal modulo del **TD-Error** ($\delta_i$):
$$p_i = |\delta_i| + \epsilon$$

Dove $\epsilon$ è una piccola costante positiva che assicura che anche le transizioni con errore nullo mantengano una probabilità minima di essere estratte. La probabilità effettiva di campionamento della transizione $i$ diventa:
$$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$$

Il parametro $\alpha$ determina l'aggressività della prioritizzazione (con $\alpha = 0$ si torna al campionamento uniforme).

**Impatto sulla Policy:** Quando l'USV affronta una curva complessa o subisce una collisione imprevista, la discrepanza tra il valore Q stimato e il reward reale genera un TD-Error elevatissimo. Il PER intercetta questa transizione e le assegna una priorità massima. Nei successivi step di ottimizzazione, la Main Network sarà costretta a estrarre e ristudiare ripetutamente quell'errore specifico, forzando i pesi della rete ad adattarsi alle geometrie critiche. Questo meccanismo elimina i punti ciechi degli spawn difficili e accelera la formazione di una policy di *collision avoidance* robusta e reattiva.

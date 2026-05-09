# Analisi Parametri e Convergenza — USV DDQN (Feng et al. 2021)

## 1. Relazione tra Epoche (Episodi), Replay Buffer e Batch Size
La domanda fondamentale è: *"La batch size di 64 è troppo piccola?"* Per capirlo, dobbiamo guardare la matematica dei dati che l'agente sta generando.

* **Epoche (3000 Episodi):** Definiscono la quantità totale di esperienza raccolta. Se in media un episodio dura 300 step prima del crash, l'agente esegue ~900,000 step totali nell'ambiente.
* **Replay Buffer (100,000):** La memoria a breve termine. L'agente ricorda solo gli ultimi 100,000 step.
* **Batch Size (64):** Ad ogni singolo step nel simulatore, l'agente estrae 64 "ricordi" a caso dal buffer per calcolare i gradienti e aggiornare la rete neurale.

### Il Problema Reale: Lo Sbilanciamento delle Classi (Diluizione dei Crash)
Consideriamo un episodio medio di 200 step che finisce contro un muro:
* **199 step** avranno reward **+5** (tutto va bene).
* **1 step** avrà reward **-1000** (collisione).
Questo significa che solo lo **0.5%** del Replay Buffer contiene l'informazione cruciale "come si muore". 

Se la tua Batch Size è **64** e usi il campionamento casuale uniforme, il numero atteso di eventi di crash in un singolo addestramento è `64 * 0.005 = 0.32`. 
**Risultato:** Nella maggior parte dei training step, l'agente non vede **NEMMENO UN CRASH**. Aggiorna i pesi vedendo solo reward di +5, dicendosi "sto andando alla grande". Poi, improvvisamente, becca un -1000, l'errore esplode e distrugge i pesi (catastrophic forgetting a micro-livello). 

## 2. Come risolve questo problema il paper di Feng?
Nel paper (Figura 3 e Sezione 3.3.2), gli autori specificano di aver ottenuto la vera stabilità usando il **PER (Prioritized Experience Replay)**. Il PER non sceglie i ricordi a caso, ma dà priorità a quelli con un errore più alto (ovvero i crash da -1000). Questo assicura che in ogni Batch di 64 ci siano *sempre* esempi di schianti.

## 3. Cosa VA e Cosa NON VA nell'implementazione attuale (Branch `feng_direct`)

### ✅ Cosa VA (Siete allineati al paper)
1.  **Architettura e Input:** Rete FC 300->300 con input 50 LIDAR bins e 11 azioni in output. Perfetto.
2.  **Reward (+5 / -1000):** Fedele all'equazione 13 del paper.
3.  **Decadimento di Epsilon (β = 0.999):** Corretto. Scendere gradualmente in 3000 episodi è ciò che ha dato i risultati migliori a Feng.
4.  **Task di Survival:** Il paper non passa le coordinate del goal alla rete, l'agente fa letteralmente *collision avoidance pura*, cercando di sopravvivere il più a lungo possibile navigando.

### ❌ Cosa NON VA (Ostacoli alla convergenza)
1.  **Uniform Replay Buffer con Batch Size 64:** Diluisce i crash. L'agente dimentica i muri.
2.  **Mean Squared Error (MSE Loss) puro:** Un reward di -1000 genera un TD-Error enorme. Elevato al quadrato dall'MSE, i gradienti sbalzano su valori assurdi (loss di 2000-3000). Questo distrugge la stabilità.
3.  **Generalizzazione Prematura:** Aspettarsi che il robot generalizzi sui Maze 1 e 3 avendo visto solo il Maze 2 con RL "puro" è impossibile. Il RL overfitta brutalmente sulla mappa geometrica di addestramento. Finché l'agente non padroneggia il Maze 2 con un success rate > 80%, è inutile testarlo sugli altri labirinti.

## 4. Stabilizzazione Matematica: Huber Loss e Gradient Clipping
Perché l'implementazione "pura" di Feng spesso fallisce senza PER? Perché il segnale di errore dello schianto è troppo violento per una rete neurale standard.

### A. Huber Loss (Smooth L1) vs MSE
La **Mean Squared Error (MSE)** eleva l'errore al quadrato: un errore di -1000 diventa una Loss di 1.000.000. Questo genera uno "shock" ai pesi della rete, causando il *Catastrophic Forgetting* (l'agente dimentica come navigare perché è terrorizzato dall'ultimo urto).

La **Huber Loss** agisce come un limitatore di velocità:
* **Errori piccoli:** Si comporta come l'MSE (quadratica), permettendo precisione millimetrica nella navigazione.
* **Errori grandi (-1000):** Diventa lineare. L'errore viene "pesato" ma non esplode esponenzialmente, permettendo alla rete di imparare che "schiantarsi è male" senza distruggere la tecnica di guida già acquisita.

### B. Gradient Clipping (Soglia 1.0)
Mentre la Loss controlla l'entità dell'errore, il **Gradient Clipping** controlla la forza dell'aggiornamento.
* **Senza Clipping:** Dopo uno schianto, i gradienti (le istruzioni di modifica per i neuroni) possono essere enormi. Se un peso della rete è `0.01` e il gradiente è `100.0`, il peso viene stravolto.
* **Con Clipping a 1.0:** Imponiamo un tetto massimo. Anche se l'errore è catastrofico, la rete non può cambiare i suoi parametri più di un certo valore per singolo step. È come dire all'IA: "Hai sbagliato molto, ma correggiti con calma, non stravolgere tutto subito".

## 5. Il Piano d'Azione Immediato (Senza stravolgere il codice)

Prima di implementare l'algoritmo complesso del PER (Prioritized Experience Replay), facciamo tre correzioni facilissime ma di impatto devastante per stabilizzare la matematica dell'addestramento:

1.  **Aumentare la Batch Size (da 64 a 256):**
    * *Perché:* Pescando 256 campioni per volta, aumenti statisticamente la probabilità che in *ogni singolo aggiornamento* della rete ci sia almeno un ricordo di una collisione (256 * 0.5% = 1.28 crash a batch). L'agente non si dimenticherà dei muri.
2.  **Passare alla Huber Loss (Smooth L1 Loss):**
    * *Perché:* L'MSE eleva l'errore al quadrato. Huber Loss eleva al quadrato gli errori piccoli (per affinare la precisione) ma è lineare per gli errori grandi (come il nostro -1000), bloccando l'esplosione dei gradienti.
3.  **Applicare il Gradient Clipping a 1.0:**
    * *Perché:* Impedisce che i gradienti distruggano l'intera rete neurale quando l'agente sbatte dopo 500 episodi di successo.

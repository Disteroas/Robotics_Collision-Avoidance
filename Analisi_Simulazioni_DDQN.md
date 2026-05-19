# Analisi Comparativa delle Run di Addestramento USV (DDQN)
## Confronto Architetturale: `matte_merge` vs `ila_merge`

Questo documento presenta un'analisi dettagliata delle performance di due diverse configurazioni dell'agente USV. Il focus è l'impatto dell'aggiunta dell'orientamento (`yaw`) al vettore di stato e la conseguente risposta dinamica dell'algoritmo di apprendimento profondo.

---

### 1. La Baseline di Riferimento (`matte_merge`)
Questa configurazione adotta un approccio conservativo e matematicamente solido, basato sulle intuizioni fondamentali del DDQN:
* **Stato:** 50 raggi LIDAR (Min-Pooling)
* **Target Network Update:** Hard Update ogni 5000 step
* **Heading (Yaw):** Assente

#### Metriche Osservate:
* **Stabilità di Apprendimento:** L'andamento è regolare. La Rete Neurale mostra un apprendimento progressivo, chiudendo il training con un `avg100` (reward medio) positivo di circa **241.3**, toccando picchi di **365.6**.
* **Crash Rate:** Sebbene globale sia del 97.2% (drogato dall'esplorazione iniziale), negli ultimi 100 episodi si è stabilizzato attorno all'**85.0%**.
* **Limiti Fisici (POMDP Aliasing):** La configurazione va incontro all'atteso overfitting locale. Punti di spawn "facili" nel Maze 2 (es. `(0.5, -2.0)` e `(0.0, 3.5)`) registrano tassi di completamento dell'episodio eccellenti (rispettivamente 60% e quasi 59%). Al contrario, nei vicoli stretti `(-1.5, -4.0)` e `(-7.0, 5.0)`, la sopravvivenza crolla al 2.7% e allo 0.3%. L'agente non comprende le variazioni d'assetto (drift) prima della collisione.
* **Generalizzazione:** Test in ambienti inediti (Maze 1 e Maze 3) confermano l'overfitting geometrico, con sopravvivenza estremamente ridotta in presenza di percorsi mai visti.

---

### 2. L'Analisi del Fallimento (`ila_merge`)
Questa configurazione parte dalla baseline ma inserisce una modifica fatale al vettore di stato:
* **Stato:** 50 raggi LIDAR + `yaw` grezzo normalizzato (odometria).

Le metriche di addestramento evidenziano un collasso sistemico istantaneo e irreversibile:
* **Collasso del Reward:** Il reward medio `avg100` precipita e si inchioda in modo permanente nell'intorno critico di **-815 / -820**. L'agente non apprende alcuna dinamica di navigazione.
* **Micro-Episodi di Schianto:** Gli episodi finali del log registrano durate irrisorie (**11, 26, 46 step**). Ciò indica che l'USV avvia immediatamente una virata a velocità massima contro i bordi del tracciato subito dopo lo spawn, palesando un disorientamento totale.
* **Esplosione della Funzione di Loss:** L'indicatore definitivo del collasso strutturale è la loss. Nelle battute finali, l'`avg_loss` schizza a valori fuori scala (fino a picchi di **8318** e oltre). In un modello DDQN stabile, questo valore deve mantenersi contenuto e privo di scosse telluriche di questa magnitudo.
* **Incapacità di Test:** Nei labirinti di validazione (Maze 3), la barca esegue un loop deterministico fallimentare (esattamente 76-77 step per episodio con reward costante a -620 e schianti garantiti).

---

### 3. Diagnosi Scientifica: Perché lo `Yaw` ha causato il collasso?
L'inserimento dello `yaw` così formattato ha generato due criticità matematiche che impediscono alla rete neurale di convergere:

#### A. La Discontinuità del Pi Greco (Gradient Explosion)
Il valore angolare di imbardata viene estratto nel range da $[-\pi, +\pi]$. Quando l'USV oscilla superando la soglia dei 180° (es. punta a ovest e devia lievemente l'assetto), l'input numerico subisce un salto repentino, passando ad esempio da $+3.14$ a $-3.14$ istantaneamente. Sebbene fisicamente l'imbarcazione sia praticamente immobile, la Rete Neurale percepisce un salto quantico nel tensore di input. Nel calcolo della backpropagation, questo gradino numerico genera gradienti colossali che "bruciano" le connessioni pre-esistenti (testimoniato dalla Loss che supera gli 8000).

#### B. La Distruzione della *Rotational Invariance*
Utilizzando l'orientamento assoluto, è stata sottratta alla rete la capacità di generalizzazione geometrica. Prima, "navigare in un corridoio dritto" richiedeva una singola reazione, indipendentemente dalla porzione di mappa. Inserendo il Nord/Sud/Est/Ovest nello stato, l'agente è costretto ad addestrarsi individualmente su come si attraversa un corridoio quando si punta a Nord, quando si punta a Est, e così via. Mappando queste direzioni assolute, la rete neurale perde i riferimenti e viene spinta in uno stato di rumore percettivo.

---

### 4. Roadmap Ingegneristica e Soluzioni per la Tesi
Per risolvere il POMDP Aliasing e abbattere il muro dell'85% di collisioni senza distruggere la backpropagation, si raccomandano tre vie operative per il prosieguo dei test:

1. **La Via Consolidata (Baseline Tesi):** Utilizzare `matte_merge` per i grafici ufficiali della tesi, documentando chiaramente le limitazioni del sensore LIDAR singolo nelle condizioni di vicolo cieco in spazi ristretti (costituisce un'ottima discussione teorica sui limiti architetturali).
2. **La Soluzione Relativa (Yaw Rate):** Se si vuole fornire l'informazione di assetto, non estrarre l'orientamento assoluto, bensì la **velocità angolare `angular.z`**. Questo rende l'input un valore relativo e continuo, informando la barca sul tasso di sbandamento momentaneo senza legarla a coordinate geografiche esterne.
3. **La Soluzione Armonica (Seno/Coseno):** Scomporre lo yaw assoluto in due componenti sempre continue: `sin(yaw)` e `cos(yaw)`. Questo incrementa il vettore di stato a 52 dimensioni ma elimina definitivamente la problematica matematica della discontinuità del $\pi$, ripristinando gradienti gestibili per il train cycle del DDQN.

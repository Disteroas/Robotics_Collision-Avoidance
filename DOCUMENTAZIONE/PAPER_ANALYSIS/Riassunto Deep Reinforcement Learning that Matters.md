# Riassunto: "Deep Reinforcement Learning that Matters" (Henderson et al., 2018)

## 1. L'Obiettivo del Paper
Il documento affronta la **crisi di riproducibilità** nel Deep Reinforcement Learning (DRL). Negli ultimi anni sono stati pubblicati algoritmi straordinari (DQN, TRPO, PPO), ma i ricercatori faticano enormemente a replicare i risultati dei paper originali. Henderson e il suo team dimostrano che le prestazioni degli algoritmi DRL sono drammaticamente influenzate da fattori "esterni" alla matematica dell'algoritmo stesso, dettagli che spesso non vengono nemmeno menzionati negli articoli scientifici.

## 2. Le 4 Cause Nascoste dell'Instabilità

Attraverso esperimenti rigorosi su ambienti standard (OpenAI Gym, MuJoCo), gli autori identificano quattro fattori critici che alterano radicalmente i risultati:

### A. La Scelta del Codebase (Libreria / Framework)
Lo stesso identico algoritmo, implementato con la stessa matematica, produce risultati diversi se scritto in TensorFlow, Theano o PyTorch. Questo accade per come le librerie calcolano i gradienti internamente, gestiscono l'ottimizzatore o applicano arrotondamenti numerici.

### B. L'Architettura della Rete Neurale
Modificare minimamente la grandezza dei layer nascosti (es. da 64x64 a 100x50) o cambiare la funzione di attivazione (da `ReLU` a `Tanh`) porta a crolli prestazionali o picchi inaspettati. Gli algoritmi DRL sono ipersensibili alla struttura della rete.

### C. La Scala del Reward
Moltiplicare i reward per una costante (es. `reward * 10`) o normalizzarli non è un'operazione neutra nel DRL. Scalare il reward influisce direttamente sulla grandezza dei gradienti e si intreccia con il Learning Rate, sballando l'apprendimento se non bilanciato.

### D. I Random Seed (Il Fattore "Fortuna")
Questa è la scoperta più devastante: **il seed casuale iniziale genera una varianza enorme**. Modificando solo il "seme" che inizializza i pesi della rete neurale o il simulatore, le prestazioni dello stesso codice possono variare dal 100% di successo al fallimento totale. I paper spesso mostrano solo le esecuzioni con i seed più fortunati.

## 3. Il Problema delle Metriche di Valutazione
Gli autori criticano aspramente come vengono presentati i grafici nei paper scientifici. Riportare il "picco massimo di reward" raggiunto durante il training o selezionare solo le 3 run migliori (cherry-picking) crea un'illusione di superiorità dell'algoritmo che non esiste nella realtà.

## 4. Raccomandazioni degli Autori
Per rendere la ricerca solida, Henderson suggerisce di:
1. Usare **più Random Seed** (almeno 5-10) per ogni esperimento.
2. Riportare **tutti gli iperparametri** utilizzati (Learning Rate, Buffer, Batch, ecc.).
3. Utilizzare **test statistici rigorosi** (es. t-test, Bootstrap) prima di affermare che un nuovo algoritmo è migliore del precedente.

---

# 🚀 LA SEZIONE PER NOI: Come usare questo paper nel tuo progetto

### 1. Il tuo scudo accademico (Per la Tesi/Esame)
Questo paper è il tuo "Jolly" per pararti le spalle. Feng **non ha pubblicato** né il codice sorgente, né il Random Seed, né gli iperparametri vitali (Target Update, Learning Rate, Pre-filling). È scientificamente provato (da Henderson) che non puoi replicare al 100% i risultati di Feng senza queste informazioni.
* **Come scriverlo nella tesi:** *"A causa dell'omissione di iperparametri chiave nel lavoro originale, e in accordo con i problemi di riproducibilità del DRL dimostrati da Henderson et al. (2018), l'implementazione ha richiesto un tuning empirico (es. Target Network Update a 1000, Buffer Pre-filling a 10000) e il confronto diretto delle curve di convergenza presenta la varianza intrinseca discussa in letteratura"*.

### 2. Smetti di stressarti se la prima run fallisce
Henderson ha dimostrato che il **Random Seed** decide la vita o la morte del training. 
* **Cosa serve a noi:** Se lanci un addestramento di 3000 episodi e il robot si schianta male, **NON buttare via il codice**. Potresti aver beccato il seed "sfortunato" (i pesi di PyTorch sono partiti malissimo). Prima di cambiare la logica o le loss, cambia il seed (`torch.manual_seed(123)`) e rilancialo.

### 3. Blindiamo l'Architettura
Il paper dice che toccare la grandezza della rete sballa tutto. 
* **Cosa serve a noi:** Nel tuo `ddqn_model.py`, **non toccare MAI** i layer `Linear(input, 300)` e `Linear(300, 300)`. Feng ha usato due layer nascosti da 300. Teniamoli identici. Più variabili togliamo dall'equazione, meno rischiamo l'instabilità descritta nel paper.

### 4. Scala del Reward e Gradienti
Nel paper, scalare il reward fa impazzire l'algoritmo.
* **Cosa serve a noi:** Noi abbiamo un reward di $+5$ e $-1000$. Questa scala l'ha decisa Feng. Noi abbiamo scoperto che questa scala, unita all'errore quadratico (MSE), genera gradienti che distruggono i pesi (Catastrophic Forgetting). Ecco perché **il Gradient Clipping a 10.0** è la nostra "pezza" fondamentale per sopravvivere a quella specifica scala di reward.

### 5. Come valutiamo i nostri test finali
Henderson distrugge l'idea di usare il "Max Reward" come metrica. E il fatto che tu usi episodi da 500 step e lui test da 5 minuti rende il "Max Reward" ancora più inutile da confrontare.
* **Cosa serve a noi:** Noi valuteremo la bontà del robot **esclusivamente con l'AVG100 (la media degli ultimi 100 episodi)** e con il *Success Rate* (% di volte che sopravvive senza sbattere). È la metrica statisticamente più robusta, approvata da questo stesso paper, e ci difenderà dalle fluttuazioni di fortuna.

# Riassunto: "Learning to Navigate in Complex Environments" (Mirowski et al., 2017 - DeepMind)

## 1. L'Obiettivo del Paper
Questo storico paper di DeepMind affronta uno dei problemi più ardui della robotica e dell'IA: la navigazione in ambienti 3D complessi (labirinti in prima persona simili al videogioco *Doom* o *Labyrinth*) partendo da input grezzi (pixel visivi) e con **reward estremamente sparsi** (l'agente riceve un premio solo quando trova il traguardo). L'obiettivo è dimostrare che un agente può imparare a orientarsi senza avere una mappa a priori (mapless navigation) e con goal che cambiano posizione di continuo.

## 2. L'Innovazione Principale: I Task Ausiliari (Auxiliary Tasks)
Il problema principale del Reinforcement Learning puro in questi ambienti è la scarsità di segnale: se l'agente riceve un premio solo dopo migliaia di step, i gradienti per addestrare la rete neurale sono inesistenti per la maggior parte del tempo. 

Per risolvere questo problema, gli autori introducono i **Task Ausiliari**. Oltre a cercare di massimizzare il reward (tramite l'algoritmo A3C), la rete neurale viene costretta a risolvere simultaneamente altri due problemi durante la navigazione:
1. **Depth Prediction (Predizione della Profondità):** L'agente deve indovinare la mappa di profondità della scena (la distanza dagli ostacoli) partendo dalla semplice immagine a colori RGB. 
2. **Loop Closure Classification (Riconoscimento dei Cicli):** L'agente deve capire se si trova in un punto del labirinto che ha già visitato in precedenza durante lo stesso episodio.

Questi task forniscono un **segnale di apprendimento continuo e denso** (dense gradient) a ogni singolo step, obbligando la rete a imparare la geometria dell'ambiente molto prima di scoprire dov'è il traguardo.

## 3. L'Architettura dell'Agente (A3C + LSTM + Ausiliari)
L'architettura proposta (chiamata *Nav A3C+D1L1*) è estremamente complessa:
* **Input Visivo:** Una Rete Neurale Convoluzionale (CNN) processa le immagini in prima persona.
* **Memoria (LSTM):** A differenza delle reti puramente reattive (Feedforward), usano una *Long Short-Term Memory* (LSTM). Questo permette all'agente di ricordarsi dove è già stato e di costruirsi una mappa mentale implicita del labirinto. 
* **Output Multiplo:** L'ultimo layer della rete non sputa fuori solo le "Azioni" e il "Valore" (come nel classico RL), ma ha delle ramificazioni che generano l'immagine di profondità e la probabilità di Loop Closure.

## 4. Risultati e Scoperta della "Mappa Interna"
I risultati sono impressionanti: l'agente supera i metodi tradizionali e, in alcuni casi, ottiene performance di navigazione pari a quelle umane. La scoperta più affascinante del paper (Sezione 5.1) è che analizzando i neuroni interni della LSTM, gli autori hanno scoperto che **l'agente ha sviluppato un "senso dell'orientamento" assoluto**: è possibile estrarre le coordinate $(X, Y)$ esatte dell'agente semplicemente guardando l'attivazione dei suoi neuroni, anche se l'agente non ha mai ricevuto queste coordinate in input.

---

# 🚀 LA SEZIONE PER NOI: Come applicare questa teoria al progetto

Questo paper di DeepMind affronta il tuo stesso identico problema teorico, ma con mezzi completamente diversi. Ecco cosa ci insegna per l'architettura su Gazebo.

### 1. La guerra ai "Reward Sparsi" (La nostra validazione)
DeepMind ha inventato i Task Ausiliari perché il loro agente prendeva punti solo alla fine del labirinto, ed era impossibile addestrarlo. Questo giustifica teoricamente tutto il lavoro fatto per evitare il problema dello schianto a 54 step. DeepMind ha risolto il problema del gradiente zero "inventando" dei task visivi continui. L'inserimento di un **Artificial Potential Field (Reward Shaping)** svolge esattamente la stessa funzione matematica: garantisce un gradiente di apprendimento denso a ogni millisecondo, evitando il collasso dell'apprendimento.

### 2. Il vantaggio sleale del LIDAR sulla CNN
Nel paper, gli autori dedicano un'enorme potenza di calcolo (CNN pesantissime) per costringere l'agente a predire la "Depth Prediction" (distanza dai muri) guardando dei pixel. 
* **Cosa significa per noi:** L'utilizzo del sensore LIDAR (i famosi 50 beam) rappresenta un bypass diretto a questo problema. L'input in virgola mobile fornisce *già* la "Depth Prediction" perfetta. Questo è il motivo accademico per cui è possibile risolvere un task di evitamento ostacoli con due semplici layer lineari da 300 neuroni (come fatto in `ddqn_model.py`), senza scomodare reti convoluzionali o task ausiliari pesanti.

### 3. Memoria LSTM vs Reti Reattive (Il dibattito sullo Stato)
DeepMind usa una LSTM (memoria temporale) perché il loro task è il *Target Reaching* (trovare un punto specifico in un labirinto dinamico). Per trovare una mela nascosta, devi ricordarti i corridoi che hai già esplorato.
* **Cosa significa per noi:** L'obiettivo primario attuale è la pura *Collision Avoidance* (evitare i muri all'infinito). Questo task obbedisce alla Proprietà di Markov: per non sbattere nel muro davanti, basta sapere a che distanza si trova in questo esatto momento, non serve sapere dov'era 10 secondi fa. L'architettura senza memoria (senza Frame Stacking e senza LSTM) è formalmente corretta e sufficiente per il task di evitamento puro.

### 4. Un'idea sperimentale: Il "LIDAR Prediction Task" (Per sviluppi futuri)
Se per la tesi si volesse dimostrare un'evoluzione dell'algoritmo base (ispirandosi direttamente a DeepMind), si potrebbe aggiungere un piccolo Task Ausiliario alla rete neurale: costringere la rete a predire quale sarà la misurazione del LIDAR allo step successivo, data l'azione attuale. Secondo il teorema esposto in questo paper, obbligare la rete a "prevedere il futuro" accelera drasticamente la sua comprensione della cinematica e dei tempi di reazione del veicolo.

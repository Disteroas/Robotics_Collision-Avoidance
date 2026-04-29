# Report del primo training - simulazione sui diversi labirinti (tratto da chat di Gemini) 

- labirinto 1: a volte riesce a "evitare di entrare nel labirinto", tuttavia quando ci entra funziona bene, per circa un minuto, poi si "freeza" e va dritto (schiantandosi)

- labirinto 2: probabilmente sono presenti curve troppo strette e si schianta sempre

 - labirinto 3: anche qui, non capisco come si schianti. Evita il primo muro e poi cerca di evitare il secondo ma gli va addosso. Una volta però è riuscito a evitarlo.

Ho paura che il training non abbia funzionato perfettamente, magari pure nel training (fatto sul labirinto 1), il robot usciva dal labirinto, restituendo il massimo reward possibile ma di fatto girando a caso. 

## Risposta di Gemini

## 🛠 Fase 1: Risoluzione del "Reward Hacking"
**Problema:** Il robot riceve il massimo reward semplicemente allontanandosi dai muri, il che lo spinge a fuggire dal labirinto verso il mare aperto.

### Azioni nel file `usv_env.py`:
1.  **Cambio della logica di Reward:** Invece di premiare il robot per essere "lontano dai muri" (distanza > 1.0m), lo premieremo per mantenere una **distanza ideale** (es. tra 0.4m e 0.8m).
2.  **Penalità di "Vuoto":** Se il LIDAR non rileva alcun ostacolo entro i 4 metri (mare aperto), il reward deve diventare neutro o leggermente negativo per forzarlo a cercare i muri del labirinto.
3.  **Reward di Avanzamento:** Premiare la velocità lineare solo quando il robot punta verso una direzione libera ma è circondato da ostacoli.

---

## 🌍 Fase 2: Addestramento Multi-Mondo (Generalizzazione)
**Problema:** Il robot impara a memoria il Labirinto 1 e fallisce nei Labirinti 2 e 3 perché non ha mai visto curve strette o angoli diversi.

### Azioni nel file `train.py` e negli script di avvio:
1.  **Rotazione Automatica dei Mondi:** Modificare il loop di addestramento affinché ogni 100 episodi lo script riavvii Gazebo caricando un labirinto diverso (`9a`, `9b`, `10`).
2.  **Punti di Partenza Random:** Inizializzare il robot in coordinate casuali all'interno del labirinto per ogni episodio, in modo che non impari un unico "percorso perfetto" ma impari a reagire agli ostacoli ovunque si trovi.

---

## 🧠 Fase 3: Frame Stacking (Memoria Temporale)
**Problema:** Il robot decide basandosi solo sull'istante attuale. Non distingue tra "stare fermo vicino a un muro" e "muoversi verso un muro".

### Azioni nel codice:
1.  **Stato Multi-Frame:** Invece di inviare alla rete neurale un array di 50 raggi LIDAR, invieremo gli ultimi **4 scan consecutivi** (un array 4x50).
2.  **Modifica Modello (`ddqn_model.py`):** Cambiare l'input della rete neurale per accettare la nuova dimensione del frame stack. Questo permetterà all'IA di percepire la **velocità relativa** degli ostacoli.

---

## ⛵ Fase 4: Fine-tuning Cinematico
**Problema:** Nei labirinti stretti, il robot è troppo veloce o ha un raggio di sterzata troppo ampio per evitare collisioni imminenti.

### Azioni nel file `usv_env.py`:
1.  **Riduzione Velocità Lineare:** Abbassare temporaneamente `LINEAR_VEL` (es. da 0.5 a 0.3) durante le fasi iniziali dell'addestramento per permettere al robot di completare le curve.
2.  **Aumento Risoluzione Azioni:** Aumentare il numero di azioni di sterzo disponibili (es. da 11 a 21) per permettere correzioni di rotta più fini e precise.

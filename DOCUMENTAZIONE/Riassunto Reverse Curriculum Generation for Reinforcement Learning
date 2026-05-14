# Riassunto: "Reverse Curriculum Generation for Reinforcement Learning" (Florensa et al., 2018)

## 1. L'Obiettivo del Paper
Questo articolo affronta i compiti "Goal-Oriented" (orientati a un obiettivo, come far uscire un robot da un labirinto o fargli inserire una chiave in una serratura). In questi compiti il *Reward è Sparso*: il robot prende un premio solo ed esclusivamente quando completa la missione. Partendo da zero, la probabilità che il robot esegua casualmente la sequenza perfetta di azioni per raggiungere l'obiettivo è praticamente nulla. Di conseguenza, il gradiente è zero e la rete non impara mai.

## 2. Il Limite del "Forward Training"
Normalmente, addestriamo i robot mettendoli nel punto di partenza (es. l'ingresso del labirinto) e sperando che trovino l'uscita. Questo richiede enormi quantità di "Reward Shaping" (dare punti intermedi, come abbiamo fatto noi con l'Artificial Potential Field) che però, se progettato male, può portare l'agente a imbrogliare (es. girare in tondo per accumulare punti senza mai uscire).

## 3. L'Innovazione: L'Addestramento al Contrario (Reverse Curriculum)
Gli autori propongono una soluzione geniale: **iniziare dalla fine**.
Il metodo funziona così:
1. **Punto di partenza vicino all'obiettivo:** Fanno nascere il robot in uno stato vicinissimo al traguardo (es. a 1 metro dall'uscita). Il robot fa un passo a caso, esce, prende il premio massimo e capisce subito cosa deve fare.
2. **Espansione a macchia d'olio:** Una volta che il robot ha imparato a vincere da quella posizione ravvicinata, l'algoritmo fa un passo indietro e genera nuovi punti di "spawn" (nascita) leggermente più lontani. 
3. **Curriculum Dinamico:** Il sistema continua a spostare i punti di partenza sempre più indietro verso l'ingresso del labirinto, ma *solo* quando il robot ha dimostrato di padroneggiare la distanza precedente.

## 4. I Risultati
L'agente impara a compiere missioni complessissime (che con il RL tradizionale falliscono al 100%) assemblando la soluzione "a ritroso". Non ha mai bisogno di esplorare ciecamente per migliaia di episodi sperando in un colpo di fortuna.

---

# 🚀 LA SEZIONE PER NOI: Come usare questo paper nel tuo progetto

Questo paper si incastra in modo perfetto con la vostra infrastruttura e vi fornisce due armi potentissime, sia per migliorare le performance che per scrivere una tesi inattaccabile.

### 1. La realizzazione del "Future Work" di Feng (Battere il paper originale)
Ricordi cosa avevamo scoperto analizzando il paper di Feng? Nelle conclusioni, Feng ammette di aver buttato il robot direttamente nel labirinto complesso (Maze 2) e scrive: *"In futuro faremo un training multi-stadio con mappe di difficoltà graduale"*. 
* **Cosa serve a noi:** Con questo paper di Florensa alla mano, voi potete letteralmente **realizzare il future work di Feng**. Potete spiegare che, per stabilizzare il training ed evitare gli schianti precoci (il famoso problema al 54° step), avete implementato una forma di *Curriculum Learning* basata sulle posizioni di spawn.

### 2. Gestione Intelligente degli `SPAWN_LISTS` (Curriculum per la Sopravvivenza)
Anche se il vostro task primario è la *Collision Avoidance* (sopravvivere) e non il *Goal Reaching* (raggiungere l'uscita), il concetto del Reverse Curriculum si applica perfettamente alla **difficoltà geometrica**.
Nel vostro file `usv_env.py` avete una lista di coordinate in cui far nascere il robot. Invece di pescarle sempre a caso fin dall'inizio, potete ordinare gli spawn per difficoltà:
1. **Livello 1 (Spazio aperto):** Fate nascere il robot in una zona larga del labirinto, lontano dai muri. Imparerà ad andare dritto.
2. **Livello 2 (Corridoio largo):** Quando l'avg100 supera i 5.000 punti, sbloccate gli spawn vicino a un singolo muro dritto.
3. **Livello 3 (Incroci a T e vicoli ciechi):** Quando l'avg100 supera i 10.000 punti, sbloccate gli spawn critici (quelli in cui sbatteva a 54 step).
* **Vantaggio:** Questo "Curriculum" di spawn geometrici garantisce che la rete impari le basi (andare dritto e schivare muri semplici) prima di dover affrontare le manovre complesse.

### 3. La carta segreta: Se il Prof chiede di "Raggiungere l'uscita"
Se a un certo punto vi viene chiesto: *"Bello che schiva i muri, ma ora fategli trovare l'uscita del labirinto"*, il RL puro fallirà miseramente. Saprete già cosa fare:
* Posizionate l'obiettivo (es. un cilindro verde) all'uscita.
* Modificate gli spawn per far nascere il robot a 2 metri dall'uscita.
* Man mano che impara, fate spostare il punto di spawn all'indietro nel labirinto, finché non impara a risolverlo partendo dall'ingresso.
* Citate *Florensa et al. (2018)* per giustificare questa scelta progettuale, dimostrando una conoscenza avanzata dello stato dell'arte accademico.

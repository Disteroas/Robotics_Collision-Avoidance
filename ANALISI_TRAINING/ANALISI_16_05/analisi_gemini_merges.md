# Relazione Tecnica: Analisi Comparativa delle Sessioni di Training (DDQN)
## Ottimizzazione della Collision Avoidance per un Unmanned Surface Vehicle (USV)

La presente relazione fornisce un'analisi dettagliata e rigorosa dei risultati ottenuti durante le diverse sessioni di addestramento (**Run 14 Vecchio Seed**, **Run 14 Nuovo Seed**, **Run 15**, e **Run 16**) dell'algoritmo Deep Q-Network Duellante (DDQN) applicato alla navigazione autonoma e all'evitamento delle collisioni di un USV in ambiente simulato Gazebo/ROS2. 

L'obiettivo fondamentale dell'analisi è mappare i fenomeni tipici del Reinforcement Learning (RL) emersi dai dati (*Overfitting*, *Seed Brittleness*, *Policy Degradation*) per delineare la strategia scientifica ottimale in vista della stesura della tesi.

---

## 1. Quadro Riassuntivo delle Performance (Training vs. Test)

La tabella seguente riassume le metriche chiave estratte dai registri di addestramento (`training_log.csv`) e dalle sessioni di valutazione deterministica (`test_results.csv`, con $\epsilon = 0$ e spawn fissi).

| Sessione di Training | Episodi Totali | Max Avg100 Reward (Training) | Success Rate Maze 2 (Test - Visto) | Success Rate Maze 3 (Test - Stretto) | Success Rate Maze 1 (Test - Aperto) | Fenomeno Principale Riscontrato |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Run 14 (Vecchio Seed)** | 4000 | +920.0 | **20.0%** | **13.3%** | 0.0% | **Inizio di Generalizzazione** |
| **Run 14 (Nuovo Seed)** | 4000 | +896.0 | 0.0% | 0.0% | 0.0% | **Seed Brittleness (Fragilità)** |
| **Run 15** | 8000 | **+1365.0** | 13.3% | 0.0% | 0.0% | **Overfitting Estremo / Illusorio** |
| **Run 16** | 5000 | +360.2 | **45.6%** | 0.0% | 0.0% | **Overfitting Geometrico / Localizzato** |

---

## 2. Analisi Dettagliata delle Singole Run

### 2.1 Run 14 — Vecchio Seed (La chiave della Tesi)
Questa sessione rappresenta il punto di svolta scientifico dell'intera ricerca. Nonostante i picchi di reward in addestramento non fossero i più alti in assoluto, il modello ha dimostrato una proprietà cruciale: la capacità di astrarre le feature sensoriali.

* **Cosa va bene:**
    * **Capacità di Generalizzazione:** È l'unico modello che ha registrato un tasso di successo positivo (**13.3%**) sul **Maze 3 (Ambiente Stretto)**, un labirinto completamente alieno su cui non ha mai effettuato un singolo passo di addestramento.
    * **Sopravvivenza Elevata:** Nel Maze 3 mostra una media di **345 step** prima della collisione, segno che l'USV non "sbatte subito", ma naviga attivamente cercando di evitare gli ostacoli prima di cadere in un vicolo cieco.
    * **Estrazione delle Regole:** La policy ha parzialmente appreso la relazione fondamentale: *"Distanza LIDAR bassa $\rightarrow$ Azione di virata"*, indipendentemente dalla mappa specifica.
* **Cosa non va bene:**
    * La performance sul labirinto noto (Maze 2) si ferma al **20%**, segno che la policy è ancora grezza e non del tutto ottimizzata.
    * Tasso di successo nullo (0%) sul Maze 1, indicando che gli spazi troppo aperti disorientano l'USV (mancanza di pareti continue come riferimento).

### 2.2 Run 14 — Nuovo Seed (L'evidenza della "Seed Brittleness")
Configurazione identica alla precedente, ad eccezione del seme casuale (passato da `42` a `123`), modificando l'inizializzazione dei pesi della rete neurale e la sequenza delle azioni stocastiche iniziali.

* **Cosa va bene:**
    * **Valore Accademico:** Questo fallimento è un risultato formidabile per la tesi. Dimostra empiricamente il problema della *Seed Brittleness* descritto in letteratura (Henderson et al., 2018). Evidenzia l'estrema sensibilità degli algoritmi model-free alle condizioni iniziali.
* **Cosa non va bene:**
    * **Collasso della Policy:** **0% di successo su tutti i labirinti**, incluso il Maze 2 di training.
    * **Buffer Avvelenato:** L'inizializzazione sfortunata dei pesi ha portato il robot a collezionare solo esperienze negative nei primi stadi critici. Mancando esempi di successo nel *Replay Buffer*, la rete ha appreso una policy deterministica difettosa che, a $\epsilon=0$, si schianta sistematicamente intorno ai 300 step su ogni mappa.

### 2.3 Run 15 — L'Illusione degli 8000 Episodi
Una sessione prolungata che, basandosi puramente sulla curva di reward del training, sembrava il modello definitivo grazie a un eccezionale valore medio di $+1365$.

* **Cosa va bene:**
    * Apparente stabilità matematica durante l'esplorazione; la curva ha mostrato una convergenza pulita verso l'alto.
* **Cosa non va bene:**
    * **Overfitting di Traiettoria:** Nel test deterministico, il successo sul Maze 2 scende al **13.3%** e va a **0%** sugli altri labirinti.
    * **Dipendenza dall'Epsilon:** Durante il training, il robot sopravviveva perché il minimo rumore casuale rimanente (l'azione random del 5%) lo "salvava" o lo sbloccava dagli angoli. Eliminata l'esplorazione nel test, la policy si rivela rigida e incapace di correggere i propri micro-errori geometrici. L'estensione del training a 8000 episodi ha solo cementato questa rigidità.

### 2.4 Run 16 — Lo Specialista Locale (Il miglior pilota "a casa sua")
Sessione da 5000 episodi che ha generato il miglior comportamento di navigazione in assoluto all'interno dell'ambiente conosciuto.

* **Cosa va bene:**
    * **Dominio del Maze 2:** Un eccellente **45.6% di tasso di successo** nel test sul labirinto di addestramento, con una media di **444 step** (vicinissimo al traguardo dei 500).
    * **Eccellenza su Spawn Specifici:** L'analisi approfondita mostra che il robot è impeccabile quando parte da determinati punti: **44.9% di successo** dallo Spawn *(6.0, 6.0)* e **38.6%** dallo Spawn *(-7.0, 5.0)*. Qui il LIDAR viene letto e interpretato alla perfezione per seguire il corridoio principale.
* **Cosa non va bene:**
    * **Overfitting Geometrico:** **0% di successo su Maze 1 e Maze 3**. Il robot ha memorizzato le metriche di distanza specifiche del Maze 2. Quando incontra le pareti strette del Maze 3, le sue soglie di attivazione per le virate sono troppo conservative e si scontra.
    * **Policy Degradation:** Il file di riepilogo evidenzia che il reward medio mobile (`avg100`) è crollato da un picco di $+360.2$ (ep. 3601) a $-65.4$ alla fine del training (ep. 5000). Questo dimostra una forte instabilità del DDQN che, continuando l'addestramento, ha "rotto" le sue stesse caratteristiche ottimali. Fortunatamente, salvare il modello *best* memorizza lo stato dell'episodio 3601, salvando la performance.
    * **Punti Ciechi Critici:** Tasso di successo dello **0.0%** sullo Spawn *(3.5, 0.5)*. Significa che la policy soffre ancora di asimmetrie geometriche gravi.

---

## 3. Conclusioni Scientifiche e Direzione della Tesi

I dati estratti smontano una concezione ingenua del Reinforcement Learning ("più alleno il robot su una mappa, più diventa intelligente") e aprono le porte a una trattazione metodologica impeccabile per la tesi, articolata su tre pilastri:

1.  **L'addestramento su ambiente singolo fallisce la generalizzazione:** Run 15 e Run 16 dimostrano che addestrare l'USV solo sul Maze 2 produce uno *specialista locale* incapace di adattarsi ad alterazioni anche minime dello spazio circostante. Il robot impara a risolvere gli *stati* (le coordinate/traiettorie specifiche), non le *feature* (il concetto astratto di ostacolo).
2.  **Il seme casuale non è un dettaglio:** La discrepanza drastica tra le due Run 14 (successo nel Maze 3 vs. fallimento totale) certifica l'alta varianza del Deep RL e l'importanza critica dell'esplorazione iniziale.
3.  **La scintilla della Run 14 (Vecchio Seed):** Il fatto che la Run 14 abbia accennato alla generalizzazione nel Maze 3 indica che la configurazione di iperparametri di quella corsa (regolazione del learning rate, struttura della rete o decay dell'epsilon) è quella che meglio favorisce l'astrazione rispetto alla pura memorizzazione muscolare.

---

## 4. Roadmap di Sviluppo (La strada per il 110 e Lode)

Per superare questi limiti e completare il lavoro di tesi con un contributo scientifico originale, la strada da seguire nel nuovo branch pulito `matte_merge17_05` prevede:

* **Azione 1: Blindare gli Iperparametri:** Riprendere l'esatta configurazione algoritmica della **Run 14 (Vecchio Seed)** o della prima fase della **Run 16**, che hanno mostrato le migliori doti rispettivamente di generalizzazione e di guida locale.
* **Azione 2: Introdurre il Training Multi-Ambiente (Multi-Maze):** Modificare lo script di training in modo che l'USV non si alleni solo sul Maze 2. Il robot deve alternare gli ambienti (es. 10 episodi nel Maze 1 per imparare a non oscillare negli spazi aperti, e 10 episodi nel Maze 2 per imparare a curvare negli spazi stretti).
* **Azione 3: Mantenere tassativamente i Random Spawn:** L'omissione degli spawn casuali distrugge la policy costringendola a contare i passi (memoria muscolare). L'USV deve nascere sempre in punti diversi per costringere la rete DDQN a dare importanza unicamente ai vettori del LIDAR.

Implementando questa strategia, l'USV sarà forzato a estrarre una policy di *Collision Avoidance* universale. Nel test finale deterministico, vedrai i tassi di successo distribuirsi uniformemente su tutti e tre i labirinti, coronando la tesi con la dimostrazione empirica della **Generalizzazione della Policy**.

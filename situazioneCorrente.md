# Situazione corrente - 29/04
Simulazione 1: fatta con un solo labirinto. Problemi di overfitting + problema nella condizione "mare apertO"

Perciò abbiamo introdotto multi maze training (branch dedicato). Il codice all'interno è autoesplicativo. Fatte due simulazioni cambiando alcuni parametri (velocità simulazione gazebo, beta dell'eps greedy,...) ma non abbiamo ottenuto grandi risultati. Il robot sopravvive a lungo nel labirinto 1, ma fa cagare negli altri. Nel labirinto 3 fa brutalmente schifo, fa sempre la stessa traiettoria

Tuttavia, anche se a livello di training andiamo male, abbiamo un buon sistema per lavorare:
- Github: diviso in repo
   - main: contiene gli script della prima simulazione, utile per runnare gazebo non in modalità headless (essenzialmente la versione base) 
   - fast_sim_claude_code: primo branch fatto per accelerare la simulazione, attualmente inutile (da chiudere)
   - old: branch di matte in cui ci sono codici bash che vanno anche in multi platform
   - multi_maze_train: attualmente il branch più avanzato. Contiene un file bash per simulazione headless multi maze in 3x che gestisce tutto in automatico. Contiene file test che in automatico switcha tra i 3 maze e produce un report di come si è comportato il robot (step medi, reward medio, ...)

Futuri updates:
- reward shaping (secondo AI questo è il problema attuale)
- capire se lo switch dei maze funzioni bene o no
- modificare usv_env in modo che ci permetta di usare la repo goddata del RL (contiene un sacco di modelli extra che potrebbero farci molto comodo)
- altri consigli di AI minori (guardare SuggestionsPostPrimoTraining)
#!/bin/bash
# =============================================================================
#  start_training_curriculum.sh  –  Orchestratore del curriculum multi-labirinto.
#
#  Questo script sostituisce start_train.sh quando si vuole allenare su
#  più labirinti. Gestisce l'intero ciclo di vita Docker da solo:
#    1. Avvia Gazebo (headless) con il labirinto del blocco corrente
#    2. Aspetta che Gazebo sia pronto
#    3. Esegue train.py per N episodi
#    4. Spegne il container
#    5. Passa al labirinto successivo e ripete
#
#  Il replay buffer viene salvato nel checkpoint dopo OGNI episodio.
#  Quando si cambia labirinto, il buffer viene ricaricato intatto:
#  la rete continua a vedere esperienze di TUTTI i labirinti precedenti,
#  prevenendo il catastrophic forgetting.
#
#  Uso:
#    ./start_training_curriculum.sh [--reset]
#
#  Flag:
#    --reset   Ignora il checkpoint e ricomincia da zero.
#              ATTENZIONE: cancella checkpoint.pkl e curriculum_state.txt
#
#  Output:
#    src/my_usv/scripts/training_log.csv       ← log episodio per episodio
#    src/my_usv/scripts/best_ddqn_model.pth    ← miglior modello
#    src/my_usv/scripts/checkpoint.pkl         ← checkpoint completo
#    src/my_usv/scripts/curriculum_state.txt   ← blocco corrente (per resume)
#    logs/sim_block_N.log                      ← log Gazebo per ogni blocco
# =============================================================================

# --------------------------------------------------------------------------- #
#  CONFIGURAZIONE CURRICULUM  ← modifica questi valori                         #
# --------------------------------------------------------------------------- #
TOTAL_EPISODES=4000          # episodi totali di training
EPISODES_PER_BLOCK=1000       # episodi per blocco prima di cambiare labirinto
GAZEBO_SPEED=3               # real_time_factor (5 = 5x più veloce del real-time)
GAZEBO_STARTUP_WAIT=30       # secondi da aspettare dopo docker run prima di exec

# Sequenza di labirinti (si ripete ciclicamente)
# MODIFICA: Utilizziamo solo 1 e 2. Il 3 resta "pulito" per i test di generalizzazione.
MAZE_SEQUENCE=(1 2)

# Percorsi interni al container
SCRIPTS_DIR_CONTAINER="/home/usv_ws/src/my_usv/scripts"
CHECKPOINT_CONTAINER="${SCRIPTS_DIR_CONTAINER}/checkpoint.pkl"

# Percorsi sul host (stesso contenuto grazie al volume mount)
SCRIPTS_DIR_HOST="$(pwd)/src/my_usv/scripts"
STATE_FILE="${SCRIPTS_DIR_HOST}/curriculum_state.txt"

# Directory per i log di Gazebo (creata automaticamente)
mkdir -p "$(pwd)/logs"

# --------------------------------------------------------------------------- #
#  CONFIGURAZIONE LABIRINTI                                                     #
# --------------------------------------------------------------------------- #
# World files (path DENTRO il container)
declare -A WORLD_PATHS
WORLD_PATHS[1]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9a.world"
WORLD_PATHS[2]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_9b.world"
WORLD_PATHS[3]="/home/usv_ws/install/my_usv/share/my_usv/worlds/labirinto_10.world"

# Coordinate di spawn (formato parametri ros2 launch)
declare -A SPAWN_COORDS
SPAWN_COORDS[1]="x:=-3 y:=-3 yaw:=1.57"
SPAWN_COORDS[2]="x:=-6 y:=0 yaw:=0"
SPAWN_COORDS[3]="x:=-2 y:=-1 yaw:=0"

# Nomi human-readable per il log
declare -A MAZE_NAMES
MAZE_NAMES[1]="Labirinto 1  (labirinto_9a)"
MAZE_NAMES[2]="Labirinto 2  (labirinto_9b)"
MAZE_NAMES[3]="Labirinto 3  (labirinto_10)"

# --------------------------------------------------------------------------- #
#  CALCOLI DERIVATI                                                             #
# --------------------------------------------------------------------------- #
TOTAL_BLOCKS=$(( (TOTAL_EPISODES + EPISODES_PER_BLOCK - 1) / EPISODES_PER_BLOCK ))
NUM_MAZES=${#MAZE_SEQUENCE[@]}

# --------------------------------------------------------------------------- #
#  FLAG --reset                                                                 #
# --------------------------------------------------------------------------- #
if [ "$1" = "--reset" ]; then
    echo ""
    echo "⚠️  Flag --reset: cancello checkpoint e stato curriculum."
    read -r -p "   Sei sicuro? (y/N) " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        rm -f "${SCRIPTS_DIR_HOST}/checkpoint.pkl" \
              "${SCRIPTS_DIR_HOST}/checkpoint.pkl.tmp" \
              "${STATE_FILE}"
        echo "   ✅ Reset completato. Ripartenza da zero."
    else
        echo "   ❌ Annullato."
        exit 0
    fi
    echo ""
fi

# --------------------------------------------------------------------------- #
#  HEADER                                                                       #
# --------------------------------------------------------------------------- #
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              USV DDQN  –  CURRICULUM MULTI-LABIRINTO                 ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  Episodi totali    : %-44s║\n" "$TOTAL_EPISODES"
printf "║  Episodi per blocco: %-44s║\n" "$EPISODES_PER_BLOCK"
printf "║  Blocchi totali    : %-44s║\n" "$TOTAL_BLOCKS"
printf "║  Sequenza labirinti: %-44s║\n" "${MAZE_SEQUENCE[*]}"
printf "║  Velocità Gazebo   : %-44s║\n" "${GAZEBO_SPEED}x real-time"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║  Log episodi : src/my_usv/scripts/training_log.csv              ║"
echo "║  Best model  : src/my_usv/scripts/best_ddqn_model.pth           ║"
echo "║  Checkpoint  : src/my_usv/scripts/checkpoint.pkl                ║"
echo "║  Log Gazebo  : logs/sim_block_N.log                             ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# --------------------------------------------------------------------------- #
#  FUNZIONI                                                                     #
# --------------------------------------------------------------------------- #

# Assicura che nessun container precedente sia in esecuzione
cleanup_container() {
    if docker inspect -f '{{.State.Running}}' usv_container &>/dev/null; then
        echo "  [Docker] Container 'usv_container' già attivo, lo termino..."
        docker stop usv_container &>/dev/null || true
        sleep 3
    fi
}

# Avvia Gazebo in background (detached) per il labirinto specificato
start_gazebo() {
    local maze_id=$1
    local block_num=$2

    local world_path="${WORLD_PATHS[$maze_id]}"
    local coords="${SPAWN_COORDS[$maze_id]}"
    local log_file="$(pwd)/logs/sim_block_${block_num}.log"
    local patched_world="/tmp/world_fast.world"

    echo "  [Gazebo] Avvio headless ${MAZE_NAMES[$maze_id]} (${GAZEBO_SPEED}x)..."
    echo "  [Gazebo] Log: logs/sim_block_${block_num}.log"

    # Avvio di Docker in background tramite Bash (& finale) anziché tramite Docker (-d)
    # Questo permette di catturare i VERI errori di ROS 2 e Python nel log
    docker run --rm --name usv_container \
        --volume="/$(pwd):/home/usv_ws" \
        usv_rl_project \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_DIR_CONTAINER}/patch_world.py \
                '${world_path}' ${GAZEBO_SPEED} ${patched_world} && \
            ros2 launch my_usv spawn_robot.launch.py \
                world:=${patched_world} \
                ${coords} \
                gui:=false
        " > "$log_file" 2>&1 &

    local docker_exit=$?
    if [ $docker_exit -ne 0 ]; then
        echo "  ❌ ERRORE: docker run ha fallito (exit code $docker_exit)."
        echo "     Controlla: docker logs usv_container"
        return 1
    fi

    echo "  [Gazebo] Container avviato. Attendo ${GAZEBO_STARTUP_WAIT}s..."
    sleep "$GAZEBO_STARTUP_WAIT"

    # Verifica che il container sia ancora vivo (Gazebo non è crashato subito)
    if ! docker inspect -f '{{.State.Running}}' usv_container 2>/dev/null | grep -q 'true'; then
        echo "  ❌ ERRORE: Gazebo è crashato durante l'avvio!"
        echo "     Ultimi log:"
        tail -30 "$log_file"
        return 1
    fi

    echo "  [Gazebo] ✅ Pronto."
    return 0
}

# Esegue train.py per un blocco di episodi
run_training_block() {
    local start_ep=$1
    local end_ep=$2
    local maze_id=$3
    local block_num=$4

    echo "  [Train] Lancio training: ep ${start_ep}+1 → ${end_ep} | ${MAZE_NAMES[$maze_id]}"

    # docker exec senza -it: blocca fino a quando train.py non esce.
    # train.py esce da solo dopo aver completato (end_ep - start_ep) episodi.
    docker exec usv_container \
        bash -c "
            cd /home/usv_ws && \
            source install/setup.bash && \
            python3 ${SCRIPTS_DIR_CONTAINER}/train.py \
                --start-ep ${start_ep} \
                --end-ep   ${end_ep} \
                --maze-id  ${maze_id} \
                --checkpoint ${CHECKPOINT_CONTAINER}
        "

    return $?
}

# Ferma il container Gazebo
stop_gazebo() {
    echo "  [Gazebo] Termino container..."
    docker stop usv_container &>/dev/null || true
    # Attendi che il container sia effettivamente rimosso (--rm nel run)
    local waited=0
    while docker inspect usv_container &>/dev/null && [ $waited -lt 15 ]; do
        sleep 1
        ((waited++))
    done
    echo "  [Gazebo] ✅ Container terminato."
}

# Handler Ctrl+C globale
_global_interrupt=false
trap '_global_interrupt=true; echo ""; echo "⚠️  Ctrl+C ricevuto. Attendo fine blocco corrente..."; stop_gazebo; exit 0' INT TERM

# --------------------------------------------------------------------------- #
#  DETERMINA IL BLOCCO DI PARTENZA (per resume)                                #
# --------------------------------------------------------------------------- #
START_BLOCK=0
if [ -f "$STATE_FILE" ]; then
    START_BLOCK=$(cat "$STATE_FILE")
    echo "📂 Curriculum state trovato: riprendo dal blocco $((START_BLOCK + 1))/$TOTAL_BLOCKS"
    echo "   (usa --reset per ripartire da zero)"
    echo ""
fi

# --------------------------------------------------------------------------- #
#  LOOP PRINCIPALE                                                              #
# --------------------------------------------------------------------------- #
total_start_time=$(date +%s)

for (( block=START_BLOCK; block<TOTAL_BLOCKS; block++ )); do

    # Calcola gli episodi globali di questo blocco
    BLOCK_START_EP=$(( block * EPISODES_PER_BLOCK ))
    BLOCK_END_EP=$(( BLOCK_START_EP + EPISODES_PER_BLOCK ))
    if [ $BLOCK_END_EP -gt $TOTAL_EPISODES ]; then
        BLOCK_END_EP=$TOTAL_EPISODES
    fi

    # Seleziona il labirinto dalla sequenza (ciclica)
    MAZE_IDX=$(( block % NUM_MAZES ))
    MAZE_ID=${MAZE_SEQUENCE[$MAZE_IDX]}

    # Calcola ETA
    elapsed=$(( $(date +%s) - total_start_time ))
    if [ $block -gt $START_BLOCK ] && [ $elapsed -gt 0 ]; then
        blocks_done=$(( block - START_BLOCK ))
        blocks_remaining=$(( TOTAL_BLOCKS - block ))
        sec_per_block=$(( elapsed / blocks_done ))
        eta_sec=$(( sec_per_block * blocks_remaining ))
        eta_str=$(printf '%dh %02dm' $((eta_sec/3600)) $((eta_sec%3600/60)))
    else
        eta_str="N/A"
    fi

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  BLOCCO $((block + 1))/$TOTAL_BLOCKS  |  ${MAZE_NAMES[$MAZE_ID]}"
    echo "  Episodi: $((BLOCK_START_EP + 1)) → $BLOCK_END_EP  |  ETA: $eta_str"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # --- Pulizia container precedente ----------------------------------------
    cleanup_container

    # --- Avvio Gazebo ---------------------------------------------------------
    if ! start_gazebo "$MAZE_ID" "$block"; then
        echo ""
        echo "❌ Gazebo non è partito per il blocco $((block + 1)). Salto e riprovo..."
        sleep 5
        continue
    fi

    # --- Training -------------------------------------------------------------
    run_training_block "$BLOCK_START_EP" "$BLOCK_END_EP" "$MAZE_ID" "$block"
    TRAIN_EXIT=$?

    if [ $TRAIN_EXIT -ne 0 ]; then
        echo ""
        echo "⚠️  train.py ha terminato con exit code $TRAIN_EXIT per il blocco $((block + 1))."
        echo "   Il checkpoint è stato salvato all'ultimo episodio completato."
        echo "   Puoi rilanciare lo script per riprendere."
    fi

    # --- Stop Gazebo ----------------------------------------------------------
    stop_gazebo
    # --- Salva stato curriculum (per resume) ----------------------------------
    # Salviamo il blocco SUCCESSIVO: se lo script viene interrotto qui,
    # al prossimo avvio ripartirà dal blocco corretto.
    echo $((block + 1)) > "$STATE_FILE"

    echo "  ✅ Blocco $((block + 1))/$TOTAL_BLOCKS completato."

    # Pausa breve tra un blocco e l'altro per stabilità
    sleep 2
done

# --------------------------------------------------------------------------- #
#  FINE CURRICULUM                                                              #
# --------------------------------------------------------------------------- #
total_elapsed=$(( $(date +%s) - total_start_time ))
total_hours=$(( total_elapsed / 3600 ))
total_min=$(( (total_elapsed % 3600) / 60 ))

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║             🏆  CURRICULUM COMPLETATO                            ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
printf "║  Episodi totali    : %-44s║\n" "$TOTAL_EPISODES"
printf "║  Tempo totale      : %-44s║\n" "${total_hours}h ${total_min}m"
printf "║  Miglior modello   : %-44s║\n" "src/my_usv/scripts/best_ddqn_model.pth"
printf "║  Log completo      : %-44s║\n" "src/my_usv/scripts/training_log.csv"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Rimuovi il file di stato: curriculum è finito, la prossima run parte da zero
rm -f "$STATE_FILE"
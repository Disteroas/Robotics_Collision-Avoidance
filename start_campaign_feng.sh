#!/bin/bash
# =============================================================================
#  start_campaign_feng.sh  —  Campagna multi-seed Feng: train + test in catena
#
#  Per ogni seed in SEEDS: lancia start_train_feng.sh, poi start_test.sh.
#  Sequenziale (i due script condividono il container 'usv_container' a nome
#  fisso → niente parallelismo). Resume-safe: senza --reset, il training
#  riprende dal checkpoint esistente.
#
#  Uso:
#    ./start_campaign_feng.sh                 # seed 0 1 2 3 4, config feng
#    ./start_campaign_feng.sh --seeds="0 1 2" # sottoinsieme
#    ./start_campaign_feng.sh --reset         # ⚠️  backup+wipe di OGNI seed
#    ./start_campaign_feng.sh --skip-test     # solo training
#
#  NB: --reset è inoltrato a start_train_feng.sh per OGNI seed (backup+wipe).
#      Ometterlo per riprendere campagne interrotte.
# =============================================================================

SEEDS="0 1 2 3 4"
CONFIG="feng"
REPS=30
RESET_FLAG=""
SKIP_TEST=0
for arg in "$@"; do
    case "$arg" in
        --seeds=*)  SEEDS="${arg#*=}" ;;
        --config=*) CONFIG="${arg#*=}" ;;
        --reps=*)   REPS="${arg#*=}" ;;
        --reset)    RESET_FLAG="--reset" ;;
        --skip-test) SKIP_TEST=1 ;;
    esac
done

mkdir -p "$(pwd)/logs"
CAMPAIGN_LOG="$(pwd)/logs/campaign_feng_$(date +%Y%m%d_%H%M%S).log"

# === Telegram bridge integration ===
# Status/control file in repo root: stessa path su Windows (Git Bash) e Python.
STATUS_FILE="$(pwd)/.cascade_status.json"
CONTROL_FILE="$(pwd)/.cascade_control"
NOTIFY="$(pwd)/notify_telegram.sh"
SEEDS_ARR=( $SEEDS )
SEEDS_TOTAL=${#SEEDS_ARR[@]}
SEEDS_DONE=0
CAMPAIGN_STARTED=$(date -Iseconds)

write_status() {
    local phase="$1" seed="$2"
    local seed_started="${3:-$(date -Iseconds)}"
    cat > "$STATUS_FILE" <<EOF
{"phase":"${phase}","seed":"${seed}","config":"${CONFIG}","started":"${CAMPAIGN_STARTED}","seed_started":"${seed_started}","log":"${CAMPAIGN_LOG}","pid":$$,"seeds_total":${SEEDS_TOTAL},"seeds_done":${SEEDS_DONE}}
EOF
}

notify() {
    [[ -x "$NOTIFY" ]] && "$NOTIFY" "$1" >/dev/null 2>&1 || true
}

check_control_between_seeds() {
    [[ -f "$CONTROL_FILE" ]] || return 0
    local ctl
    ctl=$(cat "$CONTROL_FILE" 2>/dev/null || echo "")
    if [[ "$ctl" == "abort" ]]; then
        notify "cascade ABORT received — exiting"
        return 1
    fi
    if [[ "$ctl" == "pause" ]]; then
        write_status "paused" "-"
        notify "cascade PAUSED — send /resume to continue"
        while true; do
            sleep 30
            ctl=$(cat "$CONTROL_FILE" 2>/dev/null || echo "")
            [[ "$ctl" != "pause" ]] && break
        done
        if [[ "$ctl" == "abort" ]]; then
            notify "cascade ABORT during pause — exiting"
            return 1
        fi
        notify "cascade RESUMED"
    fi
    return 0
}

: > "$CONTROL_FILE"
write_status "campaign_start" "-"
notify "campaign START config=${CONFIG} seeds=[${SEEDS}] reps=${REPS}"

trap 'notify "cascade FATAL: signal/error trap fired (rc=$?)"; write_status "error" "${SEED:-?}"' ERR

echo ""
echo "============================================================"
echo "  CAMPAGNA FENG MULTI-SEED"
echo "============================================================"
echo "  Seeds   : ${SEEDS}"
echo "  Config  : ${CONFIG}"
echo "  Reps    : ${REPS}"
echo "  Reset   : ${RESET_FLAG:-no}"
echo "  Test    : $([[ $SKIP_TEST == 1 ]] && echo no || echo si)"
echo "  Log     : ${CAMPAIGN_LOG}"
echo "============================================================"
echo ""

# Riepilogo esiti per-seed a fine campagna
declare -A TRAIN_RC TEST_RC

for SEED in $SEEDS; do
    SEED_STARTED=$(date -Iseconds)
    write_status "train" "$SEED" "$SEED_STARTED"
    notify "seed ${SEED} TRAIN start"

    echo "" | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"
    echo "  SEED ${SEED} — TRAINING  [$(date +%H:%M:%S)]"               | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"

    ./start_train_feng.sh --seed="$SEED" --config="$CONFIG" $RESET_FLAG 2>&1 | tee -a "$CAMPAIGN_LOG"
    TRAIN_RC[$SEED]=${PIPESTATUS[0]}

    if [[ "${TRAIN_RC[$SEED]}" -ne 0 ]]; then
        notify "seed ${SEED} TRAIN FAIL rc=${TRAIN_RC[$SEED]} — skip test"
        echo "  ⚠️  Train seed ${SEED} fallito (rc=${TRAIN_RC[$SEED]}). Salto il test, passo al prossimo seed." | tee -a "$CAMPAIGN_LOG"
        TEST_RC[$SEED]="skip"
        SEEDS_DONE=$((SEEDS_DONE + 1))
        check_control_between_seeds || break
        continue
    fi

    notify "seed ${SEED} TRAIN OK"

    if [[ "$SKIP_TEST" -eq 1 ]]; then
        TEST_RC[$SEED]="skip"
        SEEDS_DONE=$((SEEDS_DONE + 1))
        check_control_between_seeds || break
        continue
    fi

    write_status "test" "$SEED" "$SEED_STARTED"
    notify "seed ${SEED} TEST start"

    echo "" | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"
    echo "  SEED ${SEED} — TEST  [$(date +%H:%M:%S)]"                   | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"

    ./start_test.sh --seed="$SEED" --config="$CONFIG" --reps="$REPS" 2>&1 | tee -a "$CAMPAIGN_LOG"
    TEST_RC[$SEED]=${PIPESTATUS[0]}

    SUMMARY_FILE="runs/${CONFIG}/seed_${SEED}/eval_summary.csv"
    if [[ -f "$SUMMARY_FILE" ]]; then
        SUCC_STR=$(awk -F, 'NR>1 {printf "M%s=%.0f%% ",$1,$2*100}' "$SUMMARY_FILE")
        notify "seed ${SEED} DONE test rc=${TEST_RC[$SEED]} ${SUCC_STR}"
    else
        notify "seed ${SEED} DONE test rc=${TEST_RC[$SEED]} (no eval_summary.csv)"
    fi

    SEEDS_DONE=$((SEEDS_DONE + 1))
    check_control_between_seeds || break
done

write_status "done" "-"
SUMMARY_LINES=""
for S in $SEEDS; do
    SUMMARY_LINES="${SUMMARY_LINES}seed ${S}: train=${TRAIN_RC[$S]:-n/a} test=${TEST_RC[$S]:-n/a}; "
done
notify "campaign DONE config=${CONFIG} | ${SUMMARY_LINES}"

echo ""                                                            | tee -a "$CAMPAIGN_LOG"
echo "============================================================" | tee -a "$CAMPAIGN_LOG"
echo "  CAMPAGNA COMPLETATA — riepilogo"                           | tee -a "$CAMPAIGN_LOG"
echo "============================================================" | tee -a "$CAMPAIGN_LOG"
for SEED in $SEEDS; do
    printf "  seed %-3s | train rc=%-4s | test rc=%-4s\n" \
        "$SEED" "${TRAIN_RC[$SEED]:-n/a}" "${TEST_RC[$SEED]:-n/a}" | tee -a "$CAMPAIGN_LOG"
done
echo "============================================================" | tee -a "$CAMPAIGN_LOG"
echo "  Aggrega:  python3 src/my_usv/scripts/aggregate_seeds.py \\" | tee -a "$CAMPAIGN_LOG"
echo "    --config ${CONFIG} --output ANALISI_TRAINING/\$(date +%Y_%m_%d)/aggregate_${CONFIG}.csv" | tee -a "$CAMPAIGN_LOG"
echo "============================================================" | tee -a "$CAMPAIGN_LOG"

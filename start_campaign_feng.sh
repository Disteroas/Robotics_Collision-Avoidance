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
    echo "" | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"
    echo "  SEED ${SEED} — TRAINING  [$(date +%H:%M:%S)]"               | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"

    ./start_train_feng.sh --seed="$SEED" --config="$CONFIG" $RESET_FLAG 2>&1 | tee -a "$CAMPAIGN_LOG"
    TRAIN_RC[$SEED]=${PIPESTATUS[0]}

    if [[ "${TRAIN_RC[$SEED]}" -ne 0 ]]; then
        echo "  ⚠️  Train seed ${SEED} fallito (rc=${TRAIN_RC[$SEED]}). Salto il test, passo al prossimo seed." | tee -a "$CAMPAIGN_LOG"
        TEST_RC[$SEED]="skip"
        continue
    fi

    if [[ "$SKIP_TEST" -eq 1 ]]; then
        TEST_RC[$SEED]="skip"
        continue
    fi

    echo "" | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"
    echo "  SEED ${SEED} — TEST  [$(date +%H:%M:%S)]"                   | tee -a "$CAMPAIGN_LOG"
    echo "############################################################" | tee -a "$CAMPAIGN_LOG"

    ./start_test.sh --seed="$SEED" --config="$CONFIG" --reps="$REPS" 2>&1 | tee -a "$CAMPAIGN_LOG"
    TEST_RC[$SEED]=${PIPESTATUS[0]}
done

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

#!/bin/bash
# =============================================================================
#  notify_telegram.sh — outbound helper per notifiche Telegram.
#
#  Uso:  ./notify_telegram.sh "messaggio"
#
#  Legge .telegram_secrets accanto allo script. Fire-and-forget: errori
#  silenziati, exit code sempre 0. Mai blocca o fa fallire chi lo chiama.
# =============================================================================

MSG="${1:-}"
[[ -z "$MSG" ]] && exit 0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECRETS="${SCRIPT_DIR}/.telegram_secrets"

[[ -f "$SECRETS" ]] || exit 0

# shellcheck disable=SC1090
source "$SECRETS"

[[ -z "${TELEGRAM_BOT_TOKEN:-}" || -z "${TELEGRAM_CHAT_ID:-}" ]] && exit 0

HOST=$(hostname 2>/dev/null || echo "host")
TS=$(date '+%Y-%m-%d %H:%M:%S')
FULL_MSG="[${HOST} ${TS}]
${MSG}"

curl -s --max-time 10 \
    -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d "chat_id=${TELEGRAM_CHAT_ID}" \
    --data-urlencode "text=${FULL_MSG}" \
    >/dev/null 2>&1 || true

exit 0

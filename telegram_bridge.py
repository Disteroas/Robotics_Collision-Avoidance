#!/usr/bin/env python3
"""telegram_bridge.py — long-poll Telegram Bot API, dispatch commands.

Architecture: cooperates with start_campaign_feng.sh via two files:
  /tmp/cascade_status.json   (read by daemon)
  /tmp/cascade_control       (written by daemon)

Auth: only messages from TELEGRAM_CHAT_ID are dispatched; others ignored.
Failures are logged to stderr and the loop retries; daemon never exits
on transient errors.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
SECRETS_PATH = SCRIPT_DIR / ".telegram_secrets"
STATUS_FILE = Path("/tmp/cascade_status.json")
CONTROL_FILE = Path("/tmp/cascade_control")
LOGS_DIR = SCRIPT_DIR / "logs"
RUNS_DIR = SCRIPT_DIR / "runs"


def load_secrets(path: Path = SECRETS_PATH) -> tuple[str, int]:
    """Parse .telegram_secrets (KEY=VALUE per line, ignore # comments)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — vedi docs/telegram_bot_setup.md")
    pairs: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        pairs[k.strip()] = v.strip()
    token = pairs.get("TELEGRAM_BOT_TOKEN", "")
    chat = pairs.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        raise ValueError("TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID mancanti")
    return token, int(chat)


def send_message(token: str, chat_id: int, text: str) -> None:
    """Fire-and-forget POST to sendMessage. Never raises."""
    if not text:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text[:4000]},
            timeout=10,
        )
    except Exception as e:  # pragma: no cover (network)
        print(f"[bridge] send failed: {e}", file=sys.stderr)


def main_loop(token: str, chat_id: int) -> None:
    """Long-poll getUpdates, ack everything (dispatcher wired in Task 9)."""
    send_message(token, chat_id, "telegram bridge online")
    offset = 0
    while True:
        try:
            r = requests.get(
                f"https://api.telegram.org/bot{token}/getUpdates",
                params={"offset": offset, "timeout": 30},
                timeout=35,
            )
            data = r.json()
            for upd in data.get("result", []):
                offset = upd["update_id"] + 1
                # dispatch hook — implemented in Task 9
        except Exception as e:
            print(f"[bridge] poll failed: {e}", file=sys.stderr)
            time.sleep(5)


if __name__ == "__main__":
    try:
        TOKEN, CHAT_ID = load_secrets()
    except (FileNotFoundError, ValueError) as e:
        print(f"[bridge] startup failed: {e}", file=sys.stderr)
        sys.exit(1)
    main_loop(TOKEN, CHAT_ID)

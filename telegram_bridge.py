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
STATUS_FILE = SCRIPT_DIR / ".cascade_status.json"
CONTROL_FILE = SCRIPT_DIR / ".cascade_control"
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


def status_text(path: Path = STATUS_FILE) -> str:
    """Read cascade status snapshot; pretty multi-line summary."""
    if not path.exists():
        return "no cascade running (no status file)"
    try:
        st = json.loads(path.read_text())
    except Exception as e:
        return f"status read error: {e}"
    phase = st.get("phase", "?")
    seed = st.get("seed", "?")
    config = st.get("config", "?")
    started = st.get("started", "?")
    seed_started = st.get("seed_started", "?")
    pid = st.get("pid", "?")
    done = st.get("seeds_done", "?")
    total = st.get("seeds_total", "?")
    return (
        f"phase={phase} seed={seed} config={config}\n"
        f"campaign_started={started}\n"
        f"seed_started={seed_started}\n"
        f"progress={done}/{total} pid={pid}"
    )


def tail_text(logs_dir: Path = LOGS_DIR, pattern: str = "campaign_*.log",
              n: int = 20) -> str:
    """Last n lines of the most recent log matching pattern."""
    if not logs_dir.exists():
        return "no log (logs dir missing)"
    logs = sorted(logs_dir.glob(pattern))
    if not logs:
        return f"no log matching {pattern}"
    latest = logs[-1]
    try:
        lines = latest.read_text(errors="replace").splitlines()
    except Exception as e:
        return f"log read error: {e}"
    tail = lines[-max(1, n):]
    head = f"=== {latest.name} (last {len(tail)} of {len(lines)} lines) ===\n"
    return head + "\n".join(tail)


def seeds_text(runs_dir: Path = RUNS_DIR, config: str = "feng_hw_A") -> str:
    """Per-seed status summary for the given config."""
    cfg_dir = runs_dir / config
    if not cfg_dir.exists():
        return f"no seeds for config={config}"
    seed_dirs = sorted(
        d for d in cfg_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")
    )
    if not seed_dirs:
        return f"no seeds in {cfg_dir}"
    rows = []
    for sd in seed_dirs:
        name = sd.name
        summary = sd / "eval_summary.csv"
        if summary.exists():
            try:
                lines = summary.read_text().strip().splitlines()
                vals = []
                for ln in lines[1:]:
                    parts = ln.split(",")
                    if len(parts) >= 2:
                        vals.append(f"M{parts[0]}={float(parts[1]) * 100:.0f}%")
                rows.append(f"{name} done {' '.join(vals)}")
            except Exception as e:
                rows.append(f"{name} done (parse err: {e})")
        elif (sd / "checkpoint.pkl").exists():
            rows.append(f"{name} in-progress (checkpoint exists, no eval yet)")
        else:
            rows.append(f"{name} pending")
    return f"=== {config} ===\n" + "\n".join(rows)


def eta_text(status_path: Path = STATUS_FILE) -> str:
    """Rough ETA: extrapolate from per-seed average elapsed."""
    if not status_path.exists():
        return "no cascade running"
    try:
        st = json.loads(status_path.read_text())
    except Exception as e:
        return f"status read error: {e}"
    from datetime import datetime
    try:
        camp_start = datetime.fromisoformat(st["started"])
        done = int(st.get("seeds_done", 0))
        total = int(st.get("seeds_total", 0))
        if done < 1:
            return f"only {done} seed(s) done — ETA insufficient data"
        now = datetime.now(camp_start.tzinfo)
        elapsed = (now - camp_start).total_seconds()
        per_seed = elapsed / done
        remaining = total - done
        eta_sec = per_seed * remaining
        h = int(eta_sec // 3600)
        m = int((eta_sec % 3600) // 60)
        return (
            f"done {done}/{total} seeds in {elapsed/3600:.1f}h "
            f"(avg {per_seed/3600:.1f}h/seed)\n"
            f"ETA remaining: ~{h}h{m:02d}m"
        )
    except Exception as e:
        return f"eta compute err: {e}"


HELP_TEXT = (
    "Comandi disponibili:\n"
    "/status — fase + seed corrente + progresso\n"
    "/tail [N] — ultime N righe del log (default 20)\n"
    "/seeds — tabella per-seed (done/in-progress/pending)\n"
    "/eta — stima fine cascade\n"
    "/pause — pausa DOPO seed corrente\n"
    "/resume — annulla pausa\n"
    "/abort — kill container corrente (DANGEROUS)\n"
    "/help — questo elenco"
)


def kill_container(name: str = "usv_container") -> bool:
    """docker rm -f. Returns True if command succeeded."""
    try:
        r = subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True, timeout=30, text=True,
        )
        return r.returncode == 0
    except Exception as e:  # pragma: no cover
        print(f"[bridge] kill_container failed: {e}", file=sys.stderr)
        return False


def handle(text: str, *, config: str,
           status_path: Path = STATUS_FILE,
           logs_dir: Path = LOGS_DIR,
           runs_dir: Path = RUNS_DIR,
           control_path: Path = CONTROL_FILE,
           kill_fn=None) -> str:
    """Dispatch a command string. Returns reply text."""
    cmd = (text or "").strip()
    if cmd in ("/start", "/help"):
        return HELP_TEXT
    if cmd == "/status":
        return status_text(status_path)
    if cmd.startswith("/tail"):
        parts = cmd.split()
        n = 20
        if len(parts) > 1 and parts[1].isdigit():
            n = max(1, min(200, int(parts[1])))
        return tail_text(logs_dir=logs_dir, n=n)
    if cmd == "/seeds":
        return seeds_text(runs_dir=runs_dir, config=config)
    if cmd == "/eta":
        return eta_text(status_path)
    if cmd == "/pause":
        control_path.write_text("pause")
        return "pause requested — effective AFTER current seed completes"
    if cmd == "/resume":
        control_path.write_text("")
        return "resume — cascade will continue from next seed"
    if cmd == "/abort":
        control_path.write_text("abort")
        killer = kill_fn if kill_fn else kill_container
        ok = killer()
        suffix = "container killed" if ok else "no container running"
        return f"abort: control=abort, {suffix}. Cascade exits after current block."
    return f"unknown command: {cmd!r}\nsend /help"


def main_loop(token: str, chat_id: int, *, config: str = "feng_hw_A") -> None:
    """Long-poll getUpdates; dispatch authorized chat commands."""
    send_message(token, chat_id,
                 f"telegram bridge online — config={config}\nsend /help")
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
                msg = upd.get("message") or upd.get("edited_message")
                if not msg:
                    continue
                sender = msg.get("chat", {}).get("id")
                if sender != chat_id:
                    print(f"[bridge] ignored from chat_id={sender}",
                          file=sys.stderr)
                    continue
                text = msg.get("text", "")
                reply = handle(text, config=config)
                send_message(token, chat_id, reply)
        except Exception as e:
            print(f"[bridge] poll failed: {e}", file=sys.stderr)
            time.sleep(5)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Telegram bridge for cascade")
    p.add_argument("--config", default="feng_hw_A",
                   help="cascade config (used by /seeds). Default feng_hw_A.")
    args = p.parse_args()
    try:
        TOKEN, CHAT_ID = load_secrets()
    except (FileNotFoundError, ValueError) as e:
        print(f"[bridge] startup failed: {e}", file=sys.stderr)
        sys.exit(1)
    main_loop(TOKEN, CHAT_ID, config=args.config)

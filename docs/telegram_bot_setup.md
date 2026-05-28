# Guida — Bot Telegram per Cascade Training

Setup completo per ricevere notifiche e comandare la cascade da Telegram.
Funziona da qualsiasi rete del telefono — il PC fa solo connessioni in uscita.

## 1. Crea il bot (5 min)

1. Apri Telegram. Cerca **@BotFather** e avvia chat.
2. `/newbot`
3. Nome visualizzato (qualsiasi cosa): es. `USV Cascade`
4. Username (deve finire in `bot` ed essere univoco): es. `usv_cascade_<tuonome>_bot`
5. BotFather risponde con un **token**: `1234567890:ABCdefGhIjKlmnoPqRsTuVwXyZ`
6. **Salva il token** — è una password. Non condividerlo.

## 2. Ottieni il tuo chat_id (2 min)

1. Apri chat col bot appena creato (link in messaggio BotFather: `t.me/<username>`).
2. Manda `/start` o qualsiasi messaggio al bot.
3. In browser apri (sostituisci `<TOKEN>`):

   `https://api.telegram.org/bot<TOKEN>/getUpdates`

4. Cerca nel JSON: `"chat":{"id":12345678,"first_name":"...","type":"private"}`. Il numero è il **chat_id**.
5. Se il JSON è vuoto (`"result":[]`), manda altro messaggio al bot e ricarica.

## 3. Configura `.telegram_secrets` (1 min)

Nel root del progetto:

```bash
cp .telegram_secrets.example .telegram_secrets
```

Apri `.telegram_secrets` con editor e sostituisci con valori reali:

```
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGhIjKlmnoPqRsTuVwXyZ
TELEGRAM_CHAT_ID=12345678
```

Permessi:

```bash
chmod 600 .telegram_secrets
```

**Verifica gitignore:**

```bash
git check-ignore .telegram_secrets
```

Output atteso: `.telegram_secrets`. Se output vuoto: `.telegram_secrets` NON è gitignored — STOP, NON committare nulla.

## 4. Test notifica outbound (30s)

```bash
./notify_telegram.sh "test ping da PC"
```

Controlla Telegram: messaggio arrivato. Se no, vedi Troubleshooting in fondo.

## 5. Lancia il daemon (in terminale dedicato)

Installa `requests` se manca:

```bash
pip install requests
```

Lancia:

```bash
python3 telegram_bridge.py
```

Output atteso: log "telegram bridge online" anche nella chat Telegram.
Lascia il terminale aperto (o usa `nohup`/`screen` per staccarti).

## 6. Test comandi inbound

Dal telefono manda al bot:

| Comando | Risposta attesa |
|---|---|
| `/help` | lista comandi |
| `/status` | "no cascade running" (se cascade non attiva) |
| `/tail 5` | "no log" o ultime 5 righe se esiste log |

Se il bot non risponde: vedi Troubleshooting.

## 7. Lancia cascade con notifiche integrate

```bash
./start_campaign_feng.sh --seeds="5 6 7 8 9" --config=feng_hw_A --reps=30
```

Riceverai su Telegram (automatiche, no interazione):
- Inizio campagna
- Inizio train ogni seed
- Fine train ogni seed (con `rc`)
- Inizio test
- Fine test (con success-rate parsato da `eval_summary.csv`)
- Fine campagna (riepilogo)
- Errori al volo (se train/test crash)

Dal telefono puoi mandare:
- `/status` — fase + seed corrente
- `/tail 30` — ultime 30 righe log
- `/seeds` — tabella per-seed
- `/eta` — stima fine
- `/pause` — pausa **dopo** seed corrente
- `/resume` — riprende
- `/abort` — kill container corrente (cascade si ferma)

## Sicurezza

- **Solo il tuo `chat_id`** può comandare il bot. Chiunque altro scriva al bot viene ignorato.
- Token in `.telegram_secrets`, gitignored. Mai committare.
- `/abort` distrugge il container Docker — usalo solo se vuoi davvero fermare.
- Il daemon NON esegue comandi shell arbitrari — solo i comandi predefiniti.

## Troubleshooting

**`./notify_telegram.sh` non arriva nulla:**
1. `cat .telegram_secrets` — valori giusti?
2. `curl -s https://api.telegram.org/bot<TOKEN>/getMe` — risponde `{"ok":true,...}`?
3. `chat_id` corretto? Riprova `getUpdates`.
4. Internet outbound funzionante?

**Bot non risponde a comandi:**
1. Daemon è up? `ps aux | grep telegram_bridge.py`
2. Daemon log mostra messaggi in arrivo?
3. Stai scrivendo dallo stesso chat_id configurato?

**`/abort` non ferma cascade:**
1. Container è up? `docker ps | grep usv_container`
2. Se sì, kill manuale: `docker rm -f usv_container`
3. La cascade vede il container morto e considera il blocco fallito → passa al seed dopo o esce.

**Daemon crasha:**
1. Log su stderr nel terminale del daemon
2. Solitamente: token errato, internet down, o `.telegram_secrets` mancante
3. Riavvia con `python3 telegram_bridge.py`

## Auto-start daemon al boot (opzionale, Windows)

1. Apri **Task Scheduler**.
2. **Create Task**, nome `telegram_bridge`.
3. Trigger: **At log on**.
4. Action: **Start a program**
   - Program: `C:\Windows\System32\bash.exe` (Git Bash) o `wsl.exe`
   - Arguments: `-c "cd /c/Users/david/Desktop/PROGETTO\ ROBOTICS/Robotics_Collision-Avoidance && python3 telegram_bridge.py"`
5. **Settings → Run whether user is logged on or not**.
6. OK.

Verifica: log out + log in → daemon parte da solo.

## Disabilita / smonta

```bash
# Ferma daemon: Ctrl+C nel suo terminale
# Smonta auto-start: rimuovi task da Task Scheduler
# Cancella secrets:
rm .telegram_secrets
# (Opzionale) revoca bot:
# Telegram → @BotFather → /revoke → seleziona bot
```

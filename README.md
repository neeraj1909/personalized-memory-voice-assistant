# Personalized Memory Voice Assistant

A browser-based voice assistant that remembers your preferences and uses conversation history to give more relevant answers. Speak naturally, pause for a response, and interrupt when you want to change direction. The assistant supports English, Hindi and mixed-language conversations without a language selector.

The project is based on the MyNaksh assignment's personalized chat and memory service.

## Local setup

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/), an OpenAI API key with Realtime and chat model access, and a browser with microphone access. The app uses two local processes: the copied MyNaksh service and the Astro web gateway.

In the project directory, install dependencies and create a private `.env` file containing `OPENAI_API_KEY=your-key`. Replace `your-key` with your key; never commit this file or put the key in the browser.

```bash
cd ~/Code/conversational-ai-for-astro
uv sync
nano .env
chmod 600 .env
```

Create a private shared token once for the gateway's memory-preload requests. This preserves an existing token and does not print it:

```bash
umask 077
mkdir -p "$HOME/.local/share/conversational-ai-for-astro/local-demo"
chmod 700 "$HOME/.local/share/conversational-ai-for-astro/local-demo"
uv run python - <<'PY'
import os
import secrets
from pathlib import Path

path = Path.home() / '.local/share/conversational-ai-for-astro/local-demo/memory-warm-token'
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    pass
else:
    with os.fdopen(fd, 'w') as token_file:
        token_file.write(secrets.token_urlsafe(32) + '\n')
PY
chmod 600 "$HOME/.local/share/conversational-ai-for-astro/local-demo/memory-warm-token"
```

Start the copied MyNaksh service. These commands use the OpenAI key from `.env`, select the experimental faster single-call mode, and reuse the existing local demo memory database. Set `DATABASE_PATH` to a different private SQLite file if you want a fresh memory store. **Only one MyNaksh process may write to this database while fast mode is running**; stop any older service using the same database first.

```bash
export LLM_API_KEY="$(uv run python -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("OPENAI_API_KEY", ""))')"
export LLM_PROVIDER="openai-compatible"
export LLM_BASE_URL="https://api.openai.com/v1"
export LLM_MODEL="gpt-4.1-mini"
export ASTRO_MEMORY_FAST="1"
export DATABASE_PATH="$HOME/.local/share/conversational-ai-for-astro/local-demo/memory.sqlite3"
ASTRO_MEMORY_WARM_TOKEN="$(< "$HOME/.local/share/conversational-ai-for-astro/local-demo/memory-warm-token")"
export ASTRO_MEMORY_WARM_TOKEN

mkdir -p "$(dirname "$DATABASE_PATH")"
chmod 700 "$(dirname "$DATABASE_PATH")"
uv run uvicorn astro_memory.main:app --host 127.0.0.1 --port 8002
```

In a second terminal, start the Astro gateway. `ASTRO_WEB_AUTH_ORIGIN` must exactly match the address you open; this example uses `localhost`.

```bash
cd ~/Code/conversational-ai-for-astro
export CHAT_API_URL="http://127.0.0.1:8002/chat"
export ASTRO_WEB_AUTH_ORIGIN="http://localhost:8501"
export ASTRO_ACCOUNTS_DB="$HOME/.local/share/conversational-ai-for-astro/accounts.sqlite3"
export ASTRO_WEB_SESSIONS_DB="$HOME/.local/share/conversational-ai-for-astro/web-sessions.sqlite3"
ASTRO_MEMORY_WARM_TOKEN="$(< "$HOME/.local/share/conversational-ai-for-astro/local-demo/memory-warm-token")"
export ASTRO_MEMORY_WARM_TOKEN

mkdir -p "$(dirname "$ASTRO_ACCOUNTS_DB")"
chmod 700 "$(dirname "$ASTRO_ACCOUNTS_DB")"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8501
```

Open <http://localhost:8501>, sign in or create a local account, and click **Start talking**. Allow microphone access when the browser asks. Pause after speaking to let the assistant respond; speak again to interrupt it. Click **End conversation** when finished. Session cookies are HttpOnly, same-site, and expire after one hour. Keep both servers on loopback for local use.

## Voice behavior and cost

The browser sends audio to OpenAI Realtime over WebRTC; the Astro server handles signaling and keeps the provider key server-side. The server configures model `gpt-realtime-2.1`, transcription model `gpt-4o-mini-transcribe`, and the `marin` output voice. The UI has no manual language selector. Each voice turn calls the local MyNaksh service through `CHAT_API_URL` before the reply is spoken. Response time depends on speech detection, network conditions, both models, and audio playback; **an under-one-second spoken response is not guaranteed**.

The faster mode combines context selection and answer generation in one model call. It gives the model the signed-in user's candidate memories before relevance selection, so an irrelevant memory can influence a reply. Its multilingual and memory behavior requires further real-world validation. Set `ASTRO_MEMORY_FAST=0` to use the original two-agent flow instead. See [MyNaksh integration](astro_memory/README.md) for the copied code and its limitations.

In faster mode, the gateway preloads the authenticated user's memory and session history when they sign in, register, or restore a session. The shared token authorizes this internal request, which is sent only to a loopback chat service. If preload fails, the service loads the data when it is needed. The process-local cache is bounded and reloads evicted entries from SQLite; restarting it clears cached data, not the database. External database writers can make the cache stale, so the copied service must remain its sole writer.

The optimized and original modes can both misinterpret language preferences or recall context incorrectly. A synthetic run of the 30-turn golden dataset returned a response for every turn in both modes, but manual review found language and memory errors. The run does not establish real-time voice latency or guarantee correct answers.

Realtime API usage may be billable under the configured provider account. MyNaksh may also make separately billable model calls. Check both services' configuration, cost, and data handling before use.

## Authentication and limits

The browser UI uses a same-origin, server-owned account/session cookie. `ASTRO_WEB_AUTH_ORIGIN` enables the auth routes, realtime routes, and static `web/` UI; without it, the browser app is not mounted. The Realtime gateway binds calls to the authenticated account and session and checks that the session remains active. It limits the server to four concurrent voice sessions and each session to one hour.

The voice instructions require the model to call MyNaksh's `ask_mynaksh` tool and tell it to speak only the tool's answer. Model instructions and gateway policy checks are not a security boundary against a modified client; browser data-channel events remain client-controlled. The copied MyNaksh endpoint accepts caller-supplied user IDs and has no independent authentication. Other local processes can call it directly, bypassing browser authentication and the Realtime gateway.

Keep the app and MyNaksh services on loopback for trusted local testing. The shared `ASTRO_GATEWAY_TOKEN` and signed identity assertion still protect the legacy `/api/chat` route; they do not authenticate requests made directly to MyNaksh. Browser sign-in is not an online identity provider or production security boundary.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `ASTRO_WEB_AUTH_ORIGIN` | Exact bare browser origin; required to serve web/auth/realtime routes | Required for browser UI |
| `ASTRO_ACCOUNTS_DB` | Local account database | `~/.local/share/conversational-ai-for-astro/accounts.sqlite3` |
| `ASTRO_WEB_SESSIONS_DB` | Hashed browser-session database | `~/.local/share/conversational-ai-for-astro/web-sessions.sqlite3` |
| `CHAT_API_URL` | Full MyNaksh chat endpoint URL | Required for tool calls |
| `OPENAI_API_KEY` | Server-only Realtime credential; project `.env` takes precedence over inherited environment | Required for voice |
| `DATABASE_PATH` | SQLite memory database used by the copied MyNaksh service | `memory.db` |
| `ASTRO_MEMORY_FAST` | Set to `1` for the experimental single-call mode | Two-agent mode |
| `ASTRO_MEMORY_WARM_TOKEN` | Same private token in both server processes; enables authenticated loopback memory preload | Preload disabled if unset |
| `LLM_BASE_URL`, `LLM_API_KEY` | Model provider URL and credential for copied MyNaksh | Required for a configured chat model |
| `LLM_PROVIDER`, `LLM_MODEL` | Model provider type and model name for copied MyNaksh | `openai-compatible`, `chatgpt/gpt-6.1-sol` |

The MyNaksh endpoint accepts JSON with `user_id`, `session_id`, and `message`, and returns the existing response contract, including matching account/session IDs and a string `response`.

## Tests

```bash
uv run pytest -q
uv lock --check
(cd web && npm test)
```

Browser tests use fake media and HTTP responses; they do not make provider calls or validate real microphone/audio quality, actual MyNaksh latency, or recognition quality.

To compare the copied modes against the local golden dataset using temporary databases:

```bash
uv run python scripts/evaluate_golden.py --mode both
```

This sends synthetic dataset messages to the configured model provider and may incur API charges. It reports HTTP results, timings and exact context-key checks; answer quality, language and scope require separate review. Use `--output /private/path.json --include-responses` to save synthetic responses for that review.

## Legacy Streamlit and gateway routes

`streamlit_app.py` remains an optional older local demo; it is not the default UI. Its signed `/api/chat` flow requires `ASTRO_GATEWAY_TOKEN`, an Ed25519 private key via `ASTRO_IDENTITY_PRIVATE_KEY_FILE` in the Streamlit process, and the matching `ASTRO_IDENTITY_PUBLIC_KEY_FILE` plus `ASTRO_GATEWAY_REPLAY_DB` in the gateway process. Bearer-only `/api/chat` clients are rejected. Replay protection is per assertion, not exactly-once delivery. `/api/transcribe` remains a bearer-only optional local Whisper.cpp endpoint; its README setup and environment variables apply only to that legacy/local transcription path.

# Personalized Memory Voice Assistant

A browser-based voice assistant that remembers your preferences and uses conversation history to give more relevant answers. Speak naturally, pause for a response, and interrupt when you want to change direction. The assistant supports English, Hindi and mixed-language conversations without a language selector.

The project is based on the MyNaksh assignment's personalized chat and memory service.

## Features

- Continuous voice conversations with live transcripts and interruption.
- Per-account conversation history and long-term preferences.
- Memory preloading at sign-in for the faster response mode.
- Automatic language switching, including Hindi-English code-switching.
- A local FastAPI service and browser interface.

## Requirements

- Python 3.11 or later and [uv](https://docs.astral.sh/uv/)
- An OpenAI API key with access to Realtime and chat models
- A browser with microphone access
- Node.js to run the browser tests

OpenAI usage may incur charges. Audio and relevant conversation context are sent to the configured model provider.

## Getting started

Clone the repository and install its dependencies:

```bash
git clone https://github.com/neeraj1909/personalized-memory-voice-assistant.git
cd personalized-memory-voice-assistant
uv sync
```

Create a `.env` file in the project directory containing your API key, and keep it private:

```dotenv
OPENAI_API_KEY=your-api-key
```

Create a shared token for memory preloading. This command keeps an existing token if one is already present:

```bash
umask 077
mkdir -p "$HOME/.local/share/personalized-memory-voice-assistant"
uv run python - <<'PY'
import os
import secrets
from pathlib import Path

path = Path.home() / '.local/share/personalized-memory-voice-assistant/warm-token'
try:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
except FileExistsError:
    pass
else:
    with os.fdopen(fd, 'w') as token_file:
        token_file.write(secrets.token_urlsafe(32))
PY
```

Start the memory service:

```bash
export LLM_API_KEY="$(uv run python -c 'from dotenv import dotenv_values; print(dotenv_values(".env").get("OPENAI_API_KEY", ""))')"
export LLM_PROVIDER=openai-compatible
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=gpt-4.1-mini
export PMVA_MEMORY_FAST=1
export DATABASE_PATH="$HOME/.local/share/personalized-memory-voice-assistant/memory.sqlite3"
export PMVA_MEMORY_WARM_TOKEN="$(cat "$HOME/.local/share/personalized-memory-voice-assistant/warm-token")"
uv run uvicorn personalized_memory.main:app --host 127.0.0.1 --port 8002
```

In a second terminal, from the project directory, start the web application:

```bash
export CHAT_API_URL=http://127.0.0.1:8002/chat
export PMVA_WEB_AUTH_ORIGIN=http://localhost:8501
export PMVA_MEMORY_WARM_TOKEN="$(cat "$HOME/.local/share/personalized-memory-voice-assistant/warm-token")"
uv run uvicorn app.main:app --host 127.0.0.1 --port 8501
```

Open [http://localhost:8501](http://localhost:8501), create an account or sign in, and select **Start talking**. Allow microphone access, speak, and pause for the response. Speak again to interrupt. Select **End conversation** when finished.

Set `DATABASE_PATH` to an existing memory database to continue using it. Existing installations can continue to use their previous environment variable names; the `PMVA_` names take precedence when both are set.

## How it works

The browser connects to OpenAI Realtime over WebRTC. The web application keeps the API key on the server and sends each recognized request to the memory service, which uses saved preferences and relevant conversation context to produce a response. In fast mode, the service preloads a bounded cache and combines context selection and response generation in one model call. Use one memory-service process per database in this mode; external writes can leave its cache stale.

The memory service's `/chat` endpoint accepts `user_id`, `session_id`, and `message`. It is intended for trusted local use and does not independently authenticate callers. Keep both services on loopback. This is a learning project, not a production authentication or privacy boundary.

Responses, language selection and memory recall depend on the model and can be wrong. Response time also depends on speech detection, network conditions and model inference; subsecond voice responses are not guaranteed.

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Realtime API key, read from the project `.env` file |
| `CHAT_API_URL` | Full URL of the memory service's `/chat` endpoint |
| `PMVA_WEB_AUTH_ORIGIN` | Exact browser origin, such as `http://localhost:8501` |
| `PMVA_MEMORY_FAST` | Set to `1` for the experimental single-call mode |
| `PMVA_MEMORY_WARM_TOKEN` | Shared token used to preload memory on sign-in |
| `DATABASE_PATH` | SQLite database for conversations and memories |
| `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL` | Chat model configuration |
| `PMVA_ACCOUNTS_DB`, `PMVA_WEB_SESSIONS_DB` | Optional paths for local account and session databases |

## Tests

```bash
uv run pytest -q
uv lock --check
(cd web && npm test)
```

The automated browser tests use simulated media; they do not replace a microphone test. For an optional model-backed evaluation against a local golden dataset, run `uv run python scripts/evaluate_golden.py --mode both`. This uses temporary databases and may incur API charges.

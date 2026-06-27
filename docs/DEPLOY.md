# Deployment Guide

Deploy Memory Agent Chat locally with one command, on Streamlit Community Cloud, or with Docker.

---

## One-command local deploy (Docker)

**Prerequisites:** Docker + Docker Compose

```bash
cp .env.example .env   # add your API keys
chmod +x deploy.sh
./deploy.sh
```

Open [http://localhost:8501](http://localhost:8501)

Stop:

```bash
docker compose down
```

---

## Local development (no Docker)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in keys
streamlit run app/main.py
```

---

## Streamlit Community Cloud (live demo)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. **New app** → select repo → main file: `app/main.py`
4. Add secrets (Settings → Secrets):

```toml
GEMINI_API_KEY = "..."
DEEPSEEK_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
PINECONE_MEMORY_INDEX_NAME = "..."
```

5. Deploy — your live URL will be `https://<app-name>.streamlit.app`

Add the live URL to your README once deployed.

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GEMINI_API_KEY` | Yes | Google AI / Gemini API key |
| `DEEPSEEK_API_KEY` | Yes | DeepSeek chat model key |
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Domain knowledge index |
| `PINECONE_MEMORY_INDEX_NAME` | Yes | Memory index |
| `CHAT_DB_PATH` | No | SQLite path (default: `./chat_memory.db`) |
| `GEMINI_MULTIMODAL_MODEL` | No | Default: `gemini-2.5-flash` |

---

## Evaluation (post-deploy smoke test)

```bash
python evals/run_eval.py
python evals/run_eval.py --live-chat --output evals/results.json
```

---

## CI

GitHub Actions runs on every push/PR:

```bash
pytest tests/ -v
```

See `.github/workflows/ci.yml`.

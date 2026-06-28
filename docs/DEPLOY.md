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

### Local Google OAuth (web client JSON)

1. Place `google_client_secret.json` in the project root
2. Enable **Vertex AI API** on your GCP project
3. Run once:

```bash
python scripts/google_auth.py
```

4. This creates `google_token.json` (gitignored). Restart the app.

---

## Streamlit Community Cloud (live demo)

Streamlit Cloud **cannot read files from your repo** if they are gitignored. Put credentials in **App settings → Secrets** instead.

### Recommended: Gemini API key (no GCP org setup)

Use this when your organization blocks service account keys, OAuth is internal-only, or you want the simplest deploy:

```toml
GEMINI_API_KEY = "your-google-ai-studio-key"
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
PINECONE_MEMORY_INDEX_NAME = "..."
```

Get a key at [Google AI Studio](https://aistudio.google.com/apikey). This uses the Gemini API directly — not Vertex AI — so org IAM policies on GCP keys do not apply.

Deploy with main file: `app/main.py`

### Pre-load the sample corpus (recommended)

Visitors should be able to ask questions immediately without uploading files.

**Option A — auto-index on startup (easiest):** add to Streamlit secrets:

```toml
AUTO_INDEX_SAMPLE_CORPUS = true
```

The app indexes `sample_data/` once when the knowledge base is empty.

**Option B — index before deploy:** run locally against the same Pinecone indexes:

```bash
python scripts/index_sample_corpus.py
```

**Option C — in-app:** click **Index sample corpus** in the sidebar after the app loads.

### Deploy steps

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Repository: `noumanaziz7241/rag_streamlit` (or your fork)
4. Branch: `main`
5. Main file path: `app/main.py`
6. **Advanced settings → Secrets** — paste TOML from `.streamlit/secrets.toml.example` with real keys
7. Deploy, then add your URL to the README **Live demo** section

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=noumanaziz7241/rag_streamlit&branch=main&mainPath=app/main.py)

After deploy, smoke-test with:

- *"What tech stack does Memory Agent Chat use?"*
- *"How does MMR retrieval work?"*

---

## Sample corpus

Bundled demo files live in `sample_data/`:

| File | Topic |
|------|--------|
| `memory_agent_overview.md` | Architecture and tech stack |
| `rag_pipeline_guide.md` | Ingestion and MMR retrieval |
| `agent_tools_reference.md` | Agent tools |
| `demo_faq.md` | Demo and deploy FAQ |

Index via sidebar, CLI (`scripts/index_sample_corpus.py`), or `AUTO_INDEX_SAMPLE_CORPUS=true`.

See [ARCHITECTURE.md](./ARCHITECTURE.md) for system diagrams.

### Vertex AI: service account (only if key creation is allowed)

Skip this section if GCP shows: *"Organization Policy that blocks service account key creation"*.

1. In [Google Cloud Console](https://console.cloud.google.com/) → **IAM & Admin → Service Accounts**
2. Create a service account with **Vertex AI User** role
3. Create a JSON key and download it
4. In [share.streamlit.io](https://share.streamlit.io) → your app → **Settings → Secrets**, paste:

```toml
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
PINECONE_MEMORY_INDEX_NAME = "..."

[google_service_account]
type = "service_account"
project_id = "your-gcp-project-id"
private_key_id = "..."
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-sa@your-project.iam.gserviceaccount.com"
client_id = "..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url = "https://www.googleapis.com/robot/v1/metadata/x509/..."

GOOGLE_VERTEX_LOCATION = "us-central1"
```

5. Deploy with main file: `app/main.py`

### Alternative: OAuth client secret + token (your current JSON)

Your `google_client_secret.json` is an **OAuth web client**. It needs a **refresh token** on the server (you cannot open a browser on Streamlit Cloud).

**Step 1 — locally**, with `google_client_secret.json` in the project root:

```bash
python scripts/google_auth.py
```

**Step 2 — copy both JSON blobs into Streamlit secrets:**

```toml
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
PINECONE_MEMORY_INDEX_NAME = "..."

[google_client_secret.web]
client_id = "276270265968-....apps.googleusercontent.com"
project_id = "new-project-1-500718"
client_secret = "GOCSPX-..."
auth_uri = "https://accounts.google.com/o/oauth2/auth"
token_uri = "https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"

[google_token]
token = "ya29...."
refresh_token = "1//...."
token_uri = "https://oauth2.googleapis.com/token"
client_id = "276270265968-....apps.googleusercontent.com"
client_secret = "GOCSPX-..."
scopes = ["https://www.googleapis.com/auth/cloud-platform"]

GOOGLE_VERTEX_LOCATION = "us-central1"
```

Copy field values from:
- `google_client_secret.json` → `[google_client_secret.web]`
- `google_token.json` → `[google_token]`

Requires an OAuth consent screen that allows your account (external app + test user, or same org).

---

## DeepSeek (disabled)

DeepSeek support is **commented out** in `memory_agent/google/chat_model.py` because no API key is configured.

To re-enable later:

1. Set `DEEPSEEK_ENABLED = True` in `memory_agent/config.py`
2. Uncomment the DeepSeek block in `chat_model.py`
3. Add `DEEPSEEK_API_KEY` to secrets or `.env`

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | Yes | Pinecone API key |
| `PINECONE_INDEX_NAME` | Yes | Domain knowledge index |
| `PINECONE_MEMORY_INDEX_NAME` | Yes | Memory index |
| Google credentials | Yes* | Streamlit secrets table, local JSON, or `GEMINI_API_KEY` |
| `GOOGLE_VERTEX_LOCATION` | No | Default: `us-central1` |
| `GEMINI_CHAT_MODEL` | No | Default: `gemini-2.5-flash` |
| `DEEPSEEK_API_KEY` | No | Disabled by default |
| `CHAT_DB_PATH` | No | SQLite path (default: `./chat_memory.db`) |
| `AUTO_INDEX_SAMPLE_CORPUS` | No | Index `sample_data/` on startup when KB is empty |

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

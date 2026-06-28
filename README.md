# Memory Agent Chat

<p align="center">
  <img src="assets/demo.gif" alt="Memory Agent Chat demo — streaming responses, agent tool transparency, and source citations" width="860" />
</p>

<p align="center">
  <em>Streaming answers · expandable agent tools · grounded source citations</em>
</p>

A Streamlit chat application powered by LangGraph that combines conversational memory, domain knowledge retrieval (RAG), and persistent multi-session chat history. The agent remembers user-specific facts, retrieves chunked domain documents with MMR search, and keeps each conversation in its own checkpointed thread.

[![CI](https://github.com/noumanaziz7241/rag_streamlit/actions/workflows/ci.yml/badge.svg)](https://github.com/noumanaziz7241/rag_streamlit/actions/workflows/ci.yml)
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragapp-vjygsngajq3vodxroxft4e.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-agent-orange)](https://github.com/langchain-ai/langgraph)
[![Pinecone](https://img.shields.io/badge/Pinecone-vector%20DB-000000)](https://www.pinecone.io)

**Try it live:** [ragapp-vjygsngajq3vodxroxft4e.streamlit.app](https://ragapp-vjygsngajq3vodxroxft4e.streamlit.app/)

## Features

- **Multi-session chat** — Create, switch, clear, and delete conversations; each session has isolated history
- **Checkpoint-backed history** — LangGraph `SqliteSaver` is the source of truth; the UI reloads history from checkpoints on session switch and page refresh
- **Streaming responses** — Assistant replies stream token-by-token while the agent runs
- **Source citations** — Retrieved documents appear in an expandable **Sources** panel with filename, modality, chunk index, preview text, and image thumbnails
- **Tool transparency** — Expandable **Agent tools** panel shows when `save_memory`, `recall_memory`, or `retrieve_domain` ran and what they returned
- **Document management** — List, deduplicate, and delete indexed files from the knowledge base UI
- **Conversational memory** — Stores and recalls user-specific facts via a dedicated Pinecone memory index (filtered by user + session)
- **Multimodal RAG** — [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) indexes text, PDF, images, audio, and video in a unified vector space; [Gemini Flash](https://ai.google.dev/gemini-api/docs/models) interprets retrieved media for the chat LLM
- **Tool-augmented agent** — Gemini chat model with `save_memory`, `recall_memory`, and `retrieve_domain` tools (DeepSeek fallback available but commented out)
- **Modular codebase** — Separated into `memory_agent` (core logic) and `app` (Streamlit UI)

## Project Structure

```
rag_streamlit/
├── app/                          # Streamlit application layer
│   ├── bootstrap.py              # Environment setup, warning suppression
│   ├── main.py                   # App entry point
│   └── ui/
│       ├── chat.py               # Chat interface (streaming + citations)
│       ├── message_render.py     # Tool transparency + source citation UI
│       ├── sidebar.py            # Sessions + document upload
│       ├── documents.py          # Knowledge-base document manager
│       └── state.py              # Streamlit session state helpers
├── memory_agent/                 # Core package
│   ├── api.py                    # ChatAPI facade
│   ├── config.py                 # Constants and configuration
│   ├── models.py                 # ChatRequest, ChatResponse
│   ├── documents/
│   │   └── registry.py           # Indexed document metadata (SQLite)
│   ├── agent/
│   │   ├── graph.py              # LangGraph MemoryAgent
│   │   └── tools.py              # save_memory, recall_memory, retrieve_domain
│   ├── rag/
│   │   ├── embeddings.py         # gemini-embedding-2 client
│   │   ├── loaders.py            # All file format loaders
│   │   ├── media_store.py        # Local media persistence
│   │   ├── multimodal.py         # Gemini Flash media understanding
│   │   ├── pipeline.py           # Ingestion + MMR retrieval
│   │   └── types.py              # RagChunk dataclass
│   ├── sessions/
│   │   └── store.py              # Session metadata (SQLite)
│   └── vectorstore/
│       ├── domain_index.py       # Multimodal Pinecone index
│       └── manager.py            # Pinecone domain + memory stores
├── assets/
│   └── demo.gif                  # README demo animation
├── docs/
│   ├── ARCHITECTURE.md           # Mermaid diagrams + module map
│   ├── DEPLOY.md                 # Docker + Streamlit Cloud deployment
│   └── PORTFOLIO_ROADMAP.md      # Portfolio improvement guide
├── evals/
│   ├── corpus/                   # Fixed eval corpus
│   ├── golden_qa.json            # Golden Q&A set
│   ├── metrics.py                # recall@k + faithfulness metrics
│   └── run_eval.py               # Evaluation runner
├── sample_data/                  # Demo corpus (4 bundled markdown files)
├── scripts/
│   ├── index_sample_corpus.py    # CLI: index sample_data/ into Pinecone
│   └── generate_demo_gif.py      # Regenerate demo GIF
├── tests/                        # Pytest suite
├── .github/workflows/ci.yml      # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── deploy.sh                     # One-command Docker deploy
├── streamlit_chat.py             # Backward-compatible entry point
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── chat_memory.db                # Checkpoints + session metadata (runtime)
```

## Architecture

```mermaid
flowchart TB
    UI["Streamlit UI"] --> API["ChatAPI"]
    API --> Agent["MemoryAgent\nLangGraph"]
    Agent --> Recall["load_recalls"]
    Agent --> LLM["Gemini agent"]
    Agent --> Tools["tools"]
    Tools --> Mem["Pinecone memory"]
    Tools --> RAG["retrieve_domain"]
    RAG --> MMR["MMR + gemini-embedding-2"]
    MMR --> Domain["Pinecone domain"]
    RAG --> Flash["Gemini Flash enrichment"]
    Agent --> SQLite["SQLite checkpoints"]
```

Full diagrams (sequence flow, module map, tech decisions): **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

## Tech decisions

| Decision | Rationale |
|----------|-----------|
| **LangGraph** | Checkpointed multi-session threads, explicit tool routing, stream modes |
| **MMR retrieval** | Diverse chunks vs redundant top-k (`lambda_mult=0.65`) |
| **Gemini Embedding 2** | One embedding space for text, PDF, image, audio, and video |
| **Two Pinecone indexes** | Separate domain knowledge and session-scoped memory filters |
| **`memory_agent/` vs `app/`** | Core logic stays deployable without Streamlit (future FastAPI) |

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM | Gemini (`gemini-2.5-flash`) via Google credentials / API key |
| Embeddings | Google Gemini Embedding 2 (`gemini-embedding-2`, 768-dim) |
| Multimodal understanding | Gemini Flash (`gemini-2.5-flash`, configurable) |
| Vector store | Pinecone (serverless) |
| Session metadata + checkpoints | SQLite (`chat_memory.db`) |

## Prerequisites

- Python 3.11+
- Google access via **`GEMINI_API_KEY`** (simplest), **`google_client_secret.json`** + OAuth token locally, or Streamlit secrets on Cloud
- Pinecone API key and index names
- DeepSeek is **commented out** until an API key is available (see `memory_agent/google/chat_model.py`)

## Installation

```bash
git clone <repository-url>
cd rag_streamlit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

**Recommended (works without GCP org access):** get a key from [Google AI Studio](https://aistudio.google.com/apikey) and set `GEMINI_API_KEY` in `.env` or Streamlit secrets.

**Vertex AI (optional):** place `google_client_secret.json` in the project root (OAuth client or service account JSON from Google Cloud). Then run once:

```bash
python scripts/google_auth.py
```

This creates `google_token.json` for Vertex AI access. Enable the **Vertex AI API** on your GCP project.

> **Org-restricted GCP:** if your organization blocks service account key creation (`iam.disableServiceAccountKeyCreation`) or OAuth is internal-only, skip Vertex and use **`GEMINI_API_KEY`** only. The app falls back to the API key automatically when OAuth/Vertex credentials are incomplete.

### Streamlit Cloud deployment

On [share.streamlit.io](https://share.streamlit.io) → **Settings → Secrets**. Full guide: **[docs/DEPLOY.md](docs/DEPLOY.md)**

**Easiest — Google AI Studio API key:**

```toml
GEMINI_API_KEY = "..."
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
PINECONE_MEMORY_INDEX_NAME = "..."
```

**Vertex via service account** (only if your org allows JSON key download):

```toml
PINECONE_API_KEY = "..."
PINECONE_INDEX_NAME = "..."
PINECONE_MEMORY_INDEX_NAME = "..."

[google_service_account]
type = "service_account"
project_id = "your-gcp-project-id"
private_key = "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email = "your-sa@your-project.iam.gserviceaccount.com"
# ... remaining service account fields from downloaded JSON
```

**Your OAuth client JSON** — run `python scripts/google_auth.py` locally, then paste `[google_client_secret.web]` and `[google_token]` into Streamlit secrets (see `docs/DEPLOY.md`). Requires OAuth consent that allows your Google account.

Alternatively, create `.streamlit/secrets.toml` locally:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

```toml
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "memory-agent-domain"
PINECONE_MEMORY_INDEX_NAME = "memory-agent-memory"
GEMINI_API_KEY = "your-gemini-api-key"
```

Keys can also be set in a `.env` file as a fallback.

### Pinecone indexes

The app needs **two serverless indexes**, both **768 dimensions** (Gemini Embedding 2), **cosine** metric:

| Index env var | Purpose |
|---------------|---------|
| `PINECONE_INDEX_NAME` | Uploaded documents (RAG knowledge base) |
| `PINECONE_MEMORY_INDEX_NAME` | Long-term session memory |

Create them once with your Pinecone API key:

```bash
cp .env.example .env   # set PINECONE_API_KEY (and index names if you prefer)
python scripts/create_pinecone_indexes.py
```

Default names: `memory-agent-domain` and `memory-agent-memory`. The app also auto-creates missing indexes on first run if they do not exist.

List existing indexes: `python scripts/create_pinecone_indexes.py --list`

## Quick start (one command)

```bash
cp .env.example .env   # add API keys (GEMINI_API_KEY + Pinecone)
chmod +x deploy.sh
./deploy.sh
```

Opens at [http://localhost:8501](http://localhost:8501). Docker enables `AUTO_INDEX_SAMPLE_CORPUS` so the bundled demo docs index on first run.

See [docs/DEPLOY.md](docs/DEPLOY.md) for Streamlit Cloud and manual setup.

## Sample corpus

Four markdown files in `sample_data/` ship with the repo for instant demos:

| File | Topic |
|------|--------|
| `memory_agent_overview.md` | Architecture and tech stack |
| `rag_pipeline_guide.md` | Ingestion, MMR, tuning |
| `agent_tools_reference.md` | Memory and retrieval tools |
| `demo_faq.md` | Demo and deploy FAQ |

**Load the corpus:**

- **Sidebar** → **Index sample corpus**
- **CLI:** `python scripts/index_sample_corpus.py`
- **Docker / Cloud:** `AUTO_INDEX_SAMPLE_CORPUS=true` (default in `.env.example` and docker-compose)

## Live demo

**[Open the app →](https://ragapp-vjygsngajq3vodxroxft4e.streamlit.app/)**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ragapp-vjygsngajq3vodxroxft4e.streamlit.app/)

Pushes to `main` redeploy automatically on [Streamlit Community Cloud](https://share.streamlit.io). To run your own instance, use main file `app/main.py` and secrets from [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) (include `AUTO_INDEX_SAMPLE_CORPUS = true`). See [docs/DEPLOY.md](docs/DEPLOY.md).

Smoke-test prompts: *"What tech stack does Memory Agent Chat use?"* · *"How does MMR retrieval work?"*

## Running the App (development)

```bash
streamlit run app/main.py
```

Or via the legacy entry point:

```bash
streamlit run streamlit_chat.py
```

Opens at [http://localhost:8501](http://localhost:8501).

## Usage

### Sessions

- **New** — Start a fresh conversation thread
- **Clear** — Wipe the active session's LangGraph checkpoint
- **Delete** — Remove session metadata and its checkpoint thread

### Supported file types

| Category | Formats | How it is indexed |
|----------|---------|-------------------|
| Text / code | `.txt`, `.md`, `.csv`, `.json`, `.py`, `.html`, … | Chunked text → `gemini-embedding-2` |
| PDF | `.pdf` | Native PDF embedding (6 pages per chunk) |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.gif` | Native image embedding |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac`, `.ogg` | Native audio embedding (~180s limit) |
| Video | `.mp4`, `.mov`, `.webm`, `.avi` | Native video embedding (~120s limit) |
| Office | `.docx`, `.xlsx`, `.pptx` | Text extraction → chunked embedding |

Retrieved image/audio/video/PDF chunks are described at answer time with **Gemini Flash** so the text-based chat LLM can use them.

> **Note:** `gemini-embedding-2` uses a different vector space than `gemini-embedding-001`. Re-index existing Pinecone data after upgrading.

### Documents

Upload any supported file in the sidebar and click **Index documents**. When you ask a question, the assistant cites retrieved chunks under **Sources** in each reply.

### Demo flow

1. Click **Index sample corpus** in the sidebar (or use auto-index on Docker/Cloud)
2. Ask: *"What tech stack does Memory Agent Chat use?"*
3. Watch the response **stream** in real time
4. Expand **Agent tools** to see `retrieve_domain` (and memory tools when used)
5. Expand **References** — inline `[1]`, `[2]` should match the numbered list

Example prompts:

- *"How does MMR retrieval work in this project?"*
- *"What tools does the agent have for memory and retrieval?"*
- *"Remember that my favorite programming language is Python."* → then *"What do you remember about me?"*

### Memory

Tell the agent facts to remember. Memories are scoped to the current user and session.

### Regenerate the demo GIF

The hero GIF is checked in at `assets/demo.gif`. To recreate it (or replace with a real screen recording converted to GIF):

```bash
python scripts/generate_demo_gif.py
```

For a live screen capture, record the app with [Peek](https://github.com/phw/peek) or [LICEcap](https://www.cockos.com/licecap/), then save as `assets/demo.gif`.

## Testing & evaluation

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
python evals/run_eval.py
python evals/run_eval.py --live-chat --output evals/results.json
```

CI runs automatically on push/PR via GitHub Actions.

## Key Modules

| Module | Responsibility |
|--------|----------------|
| `app/main.py` | Streamlit entry point and page layout |
| `app/ui/chat.py` | Streaming chat UI with tool and citation panels |
| `app/ui/message_render.py` | Renders agent tool activity and source citations |
| `memory_agent/api.py` | Sessions, history, ingestion, streaming chat API |
| `memory_agent/agent/graph.py` | LangGraph agent with streaming events + checkpointed threads |
| `memory_agent/rag/embeddings.py` | `gemini-embedding-2` embedding client |
| `memory_agent/rag/loaders.py` | Multimodal file loaders |
| `memory_agent/rag/multimodal.py` | Gemini Flash media descriptions |
| `memory_agent/vectorstore/domain_index.py` | Multimodal Pinecone index + MMR |

## License

See the repository for license information.

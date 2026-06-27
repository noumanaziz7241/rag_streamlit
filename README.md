# Memory Agent Chat

<p align="center">
  <img src="assets/demo.gif" alt="Memory Agent Chat demo — streaming responses, agent tool transparency, and source citations" width="860" />
</p>

<p align="center">
  <em>Streaming answers · expandable agent tools · grounded source citations</em>
</p>

A Streamlit chat application powered by LangGraph that combines conversational memory, domain knowledge retrieval (RAG), and persistent multi-session chat history. The agent remembers user-specific facts, retrieves chunked domain documents with MMR search, and keeps each conversation in its own checkpointed thread.

[![CI](https://github.com/noumanaziz7241/rag_streamlit/actions/workflows/ci.yml/badge.svg)](https://github.com/noumanaziz7241/rag_streamlit/actions/workflows/ci.yml)

## Features

- **Multi-session chat** — Create, switch, clear, and delete conversations; each session has isolated history
- **Checkpoint-backed history** — LangGraph `SqliteSaver` is the source of truth; the UI reloads history from checkpoints on session switch and page refresh
- **Streaming responses** — Assistant replies stream token-by-token while the agent runs
- **Source citations** — Retrieved documents appear in an expandable **Sources** panel with filename, modality, chunk index, preview text, and image thumbnails
- **Tool transparency** — Expandable **Agent tools** panel shows when `save_memory`, `recall_memory`, or `retrieve_domain` ran and what they returned
- **Document management** — List, deduplicate, and delete indexed files from the knowledge base UI
- **Conversational memory** — Stores and recalls user-specific facts via a dedicated Pinecone memory index (filtered by user + session)
- **Multimodal RAG** — [Gemini Embedding 2](https://ai.google.dev/gemini-api/docs/models/gemini-embedding-2) indexes text, PDF, images, audio, and video in a unified vector space; [Gemini Flash](https://ai.google.dev/gemini-api/docs/models) interprets retrieved media for the chat LLM
- **Tool-augmented agent** — DeepSeek model with `save_memory`, `recall_memory`, and `retrieve_domain` tools
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
│   ├── DEPLOY.md                 # Docker + Streamlit Cloud deployment
│   └── PORTFOLIO_ROADMAP.md       # Portfolio improvement guide
├── evals/
│   ├── corpus/                   # Fixed eval corpus
│   ├── golden_qa.json            # Golden Q&A set
│   ├── metrics.py                # recall@k + faithfulness metrics
│   └── run_eval.py               # Evaluation runner
├── sample_data/                  # Demo files for live showcase
├── tests/                        # Pytest suite
├── .github/workflows/ci.yml      # GitHub Actions CI
├── Dockerfile
├── docker-compose.yml
├── deploy.sh                     # One-command Docker deploy
├── scripts/
│   └── generate_demo_gif.py      # Regenerate demo GIF
├── streamlit_chat.py             # Backward-compatible entry point
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── chat_memory.db                # Checkpoints + session metadata (runtime)
```

## Architecture

```
Streamlit UI (app/)
        │
        ▼
   ChatAPI (memory_agent/api.py)
        │
        ├── SessionStore ──► SQLite session metadata
        └── MemoryAgent ──► LangGraph checkpointed threads
                │
                ├── load_recalls ──► recall_memory (Pinecone memory)
                ├── agent ──► DeepSeek + full thread history
                └── tools ──► save_memory | recall_memory | retrieve_domain
                                      │
                                      ▼
                            rag/pipeline + domain_index (multimodal MMR)
                                      │
                                      ▼
              gemini-embedding-2 + Gemini Flash + Pinecone
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Agent orchestration | LangGraph |
| LLM | DeepSeek (`deepseek-chat`) |
| Embeddings | Google Gemini Embedding 2 (`gemini-embedding-2`, 768-dim) |
| Multimodal understanding | Gemini Flash (`gemini-2.5-flash`, configurable) |
| Vector store | Pinecone (serverless) |
| Session metadata + checkpoints | SQLite (`chat_memory.db`) |

## Prerequisites

- Python 3.11+
- API keys for [DeepSeek](https://platform.deepseek.com/), [Google AI (Gemini)](https://aistudio.google.com/), and [Pinecone](https://www.pinecone.io/)

## Installation

```bash
git clone <repository-url>
cd rag_streamlit
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Configuration

Create `.streamlit/secrets.toml` (see `.env.example`):

```toml
GEMINI_API_KEY = "your-gemini-api-key"
DEEPSEEK_API_KEY = "your-deepseek-api-key"
PINECONE_API_KEY = "your-pinecone-api-key"
PINECONE_INDEX_NAME = "your-domain-index-name"
PINECONE_MEMORY_INDEX_NAME = "your-memory-index-name"
```

Keys can also be set in a `.env` file as a fallback.

## Quick start (one command)

```bash
cp .env.example .env   # add API keys
chmod +x deploy.sh
./deploy.sh
```

Opens at [http://localhost:8501](http://localhost:8501). See [docs/DEPLOY.md](docs/DEPLOY.md) for Streamlit Cloud and manual setup.

## Live demo

Deploy to [Streamlit Community Cloud](https://share.streamlit.io) with main file `app/main.py` and secrets from `.streamlit/secrets.toml.example`. Add your live URL here after deploying:

```
https://YOUR-APP-NAME.streamlit.app
```

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

1. Upload a PDF or text file in the sidebar and click **Index documents**
2. Ask a question about the uploaded content
3. Watch the response **stream** in real time
4. Expand **Agent tools** to see `retrieve_domain` (and memory tools when used)
5. Expand **Sources** to inspect filenames, chunk indexes, and previews

Example prompts:

- *"Summarize the main points from the uploaded document."*
- *"Remember that my favorite programming language is Python."* → then *"What do you remember about me?"*
- *"What does the image in my knowledge base show?"* (after uploading an image)

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

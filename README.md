# Memory Agent Chat

A Streamlit chat application powered by LangGraph that combines conversational memory, domain knowledge retrieval (RAG), and persistent multi-session chat history. The agent remembers user-specific facts, retrieves chunked domain documents with MMR search, and keeps each conversation in its own checkpointed thread.

## Features

- **Multi-session chat** — Create, switch, clear, and delete conversations; each session has isolated history
- **Checkpoint-backed history** — LangGraph `SqliteSaver` is the source of truth; the UI reloads history from checkpoints on session switch and page refresh
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
│       ├── chat.py               # Chat interface
│       ├── sidebar.py            # Sessions + document upload
│       └── state.py              # Streamlit session state helpers
├── memory_agent/                 # Core package
│   ├── api.py                    # ChatAPI facade
│   ├── config.py                 # Constants and configuration
│   ├── models.py                 # ChatRequest, ChatResponse
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
├── streamlit_chat.py             # Backward-compatible entry point
├── requirements.txt
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

## Running the App

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

Upload any supported file in the sidebar and click **Index documents**.

### Memory

Tell the agent facts to remember. Memories are scoped to the current user and session.

## Key Modules

| Module | Responsibility |
|--------|----------------|
| `app/main.py` | Streamlit entry point and page layout |
| `memory_agent/api.py` | Sessions, history, ingestion, and chat API |
| `memory_agent/agent/graph.py` | LangGraph agent with checkpointed threads |
| `memory_agent/rag/embeddings.py` | `gemini-embedding-2` embedding client |
| `memory_agent/rag/loaders.py` | Multimodal file loaders |
| `memory_agent/rag/multimodal.py` | Gemini Flash media descriptions |
| `memory_agent/vectorstore/domain_index.py` | Multimodal Pinecone index + MMR |

## License

See the repository for license information.

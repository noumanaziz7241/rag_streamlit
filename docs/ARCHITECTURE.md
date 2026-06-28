# Architecture

Memory Agent Chat separates the Streamlit UI from a LangGraph agent core with multimodal RAG and checkpointed sessions.

## System overview

```mermaid
flowchart TB
    subgraph UI["Streamlit UI (app/)"]
        Sidebar["Sidebar\nsessions · upload · sample corpus"]
        Chat["Chat\nstreaming · tools · references"]
        Docs["Document manager"]
    end

    subgraph API["ChatAPI (memory_agent/api.py)"]
        Sessions["SessionStore\nSQLite metadata"]
        Agent["MemoryAgent\nLangGraph"]
    end

    subgraph Graph["LangGraph agent"]
        Recall["load_recalls\npreload memories"]
        LLM["agent\nGemini + tools"]
        Tools["tools\nToolNode"]
        Recall --> LLM
        LLM -->|tool calls| Tools
        Tools --> LLM
        LLM -->|final reply| Chat
    end

    subgraph RAG["Multimodal RAG"]
        Loaders["loaders.py\nchunk by modality"]
        Embed["gemini-embedding-2"]
        MMR["domain_index.py\nMMR search"]
        Flash["Gemini Flash\nmedia enrichment"]
    end

    subgraph Stores["Persistence"]
        PineDomain["Pinecone\ndomain index"]
        PineMem["Pinecone\nmemory index"]
        SQLite["SQLite\ncheckpoints + registry"]
    end

    Sidebar --> API
    Chat --> API
    Docs --> API
    API --> Sessions
    API --> Agent
    Agent --> Graph
    Tools -->|retrieve_domain| MMR
    Tools -->|save/recall memory| PineMem
    MMR --> Flash
    MMR --> PineDomain
    Loaders --> Embed --> PineDomain
    Agent --> SQLite
    Sessions --> SQLite
```

## Request lifecycle (one chat turn)

```mermaid
sequenceDiagram
    participant U as User
    participant UI as Streamlit chat
    participant API as ChatAPI
    participant G as LangGraph
    participant M as Pinecone memory
    participant D as Pinecone domain
    participant L as Gemini Flash

    U->>UI: message
    UI->>API: chat_stream()
    API->>G: stream graph

    G->>M: recall_memory (preload)
    G-->>UI: status / agent tools

    alt domain question
        G->>D: retrieve_domain (MMR)
        D-->>G: ranked chunks
        G->>L: enrich PDF/image/audio
        L-->>G: text descriptions
        G-->>UI: source events [1][2]…
    end

    G->>G: Gemini answer (inline citations)
    G-->>UI: token stream
    G-->>UI: done + tools + sources

    Note over G,SQLite: SqliteSaver persists full thread
```

## Module map

| Layer | Path | Role |
|-------|------|------|
| Entry | `app/main.py` | Page config, bootstrap, auto-index sample corpus |
| UI | `app/ui/chat.py` | Streaming chat, empty state, citation panels |
| UI | `app/ui/sidebar.py` | Sessions, upload, **Index sample corpus** |
| Facade | `memory_agent/api.py` | Sessions, history, ingest, streaming |
| Agent | `memory_agent/agent/graph.py` | LangGraph nodes, stream events, citations |
| Tools | `memory_agent/agent/tools.py` | `save_memory`, `recall_memory`, `retrieve_domain` |
| RAG | `memory_agent/rag/pipeline.py` | Ingest + retrieval constants |
| RAG | `memory_agent/rag/loaders.py` | Multimodal chunking |
| Vector | `memory_agent/vectorstore/domain_index.py` | Pinecone upsert + MMR |
| Demo | `memory_agent/demo/corpus.py` | Bundled `sample_data/` indexing |
| Demo | `scripts/index_sample_corpus.py` | CLI corpus loader |

## Tech decisions

| Choice | Why |
|--------|-----|
| **LangGraph** | Checkpointed multi-session threads, explicit tool routing, stream modes |
| **Gemini Embedding 2** | Single embedding space for text, PDF, image, audio, video |
| **MMR (`lambda=0.65`)** | Diverse context vs pure top-k — reduces redundant chunks |
| **Two Pinecone indexes** | Domain knowledge vs session-scoped memory with different filters |
| **SQLite checkpoints** | Durable conversation state without a separate DB for demos |
| **Streamlit** | Fast portfolio UI; core logic stays in `memory_agent/` for future FastAPI |

## Sample corpus & live demo

Bundled files live in `sample_data/`. For public demos:

1. Deploy to Streamlit Cloud (`app/main.py`)
2. Set secrets (see [DEPLOY.md](./DEPLOY.md))
3. Enable `AUTO_INDEX_SAMPLE_CORPUS=true` so first visitors get a pre-loaded knowledge base

See also [PORTFOLIO_ROADMAP.md](./PORTFOLIO_ROADMAP.md) for planned enhancements.

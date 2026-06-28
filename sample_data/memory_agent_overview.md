# Memory Agent Chat — Overview

Memory Agent Chat is a multimodal RAG application built with LangGraph, Streamlit, Pinecone, and Gemini Embedding 2.

## Architecture

- **UI layer**: Streamlit (`app/`)
- **Agent**: LangGraph with checkpointed sessions, memory tools, and domain retrieval
- **RAG**: Multimodal ingestion (text, PDF, images, audio, video, Office) with MMR retrieval
- **Vector store**: Pinecone serverless indexes for domain knowledge and session memory
- **Persistence**: SQLite for session metadata, document registry, and LangGraph checkpoints

## Key features

1. **Multi-session chat** — isolated conversation threads with persistent history
2. **Conversational memory** — user facts stored in a dedicated Pinecone memory index
3. **Multimodal RAG** — unified embedding space via `gemini-embedding-2`
4. **Streaming responses** — token streaming with tool transparency and source citations
5. **Document management** — upload, list, deduplicate, and delete indexed documents

## Tech stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Agent | LangGraph + Gemini 2.5 Flash |
| Embeddings | Gemini Embedding 2 |
| Multimodal | Gemini Flash |
| Vector DB | Pinecone |
| Checkpoints | SQLite |

## Demo questions

- What architecture does this project use?
- What file types can be indexed?
- How does the agent handle memory and retrieval?

# RAG Pipeline Guide

This document describes how Memory Agent Chat ingests and retrieves domain knowledge.

## Ingestion flow

1. **Upload** — Files arrive via the Streamlit sidebar or the sample corpus loader.
2. **Load** — `memory_agent/rag/loaders.py` detects modality (text, PDF, image, audio, video, Office).
3. **Chunk** — Text is split (~1000 chars, 200 overlap). PDFs use 6 pages per chunk. Media files become single chunks.
4. **Embed** — Each chunk is embedded with **Gemini Embedding 2** (`gemini-embedding-2`, 768 dimensions).
5. **Store** — Vectors and metadata land in the Pinecone **domain index** under namespace `polaris`.

## Retrieval flow

1. User asks a question; the agent may call **`retrieve_domain`**.
2. The query is embedded with the same model.
3. Pinecone returns top candidates; **MMR** (Maximal Marginal Relevance) selects diverse chunks (`lambda_mult=0.65`, default `k=6`).
4. Non-text chunks are enriched with **Gemini Flash** descriptions before the chat model sees them.
5. Retrieved blocks are numbered `[1]`, `[2]`, … for inline citations in the answer.

## Tuning parameters

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `RETRIEVAL_K` | 6 | Chunks returned to the agent |
| `RETRIEVAL_FETCH_K` | 20 | Candidate pool before MMR |
| `MMR_LAMBDA` | 0.65 | Balance relevance vs diversity |
| `MIN_RELEVANCE_SCORE` | 0.40 | Drop weak matches |

## Supported modalities

- **Text / code** — `.txt`, `.md`, `.py`, `.json`, `.csv`, and similar
- **PDF** — native PDF embedding, 6 pages per chunk
- **Images** — `.png`, `.jpg`, `.webp`, `.gif`
- **Audio** — `.mp3`, `.wav`, `.m4a` (up to ~180 seconds)
- **Video** — `.mp4`, `.mov`, `.webm` (up to ~120 seconds)
- **Office** — `.docx`, `.xlsx`, `.pptx` (text extracted then chunked)

## Demo questions

- How does MMR retrieval work in this project?
- What embedding model is used for indexing?
- How many chunks does retrieval return by default?

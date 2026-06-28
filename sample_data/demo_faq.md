# Memory Agent Chat — Demo FAQ

Quick answers for reviewers trying the live demo or local Docker deploy.

## Getting started

**Q: How do I try the app without uploading my own files?**  
A: Click **Index sample corpus** in the sidebar, or run `python scripts/index_sample_corpus.py` before starting the app.

**Q: What should I ask after indexing the sample corpus?**  
A: See the suggested prompts in the chat empty state, or ask about the tech stack, RAG pipeline, or agent tools.

## Architecture

**Q: What orchestrates the agent?**  
A: **LangGraph** with checkpointed threads (`SqliteSaver`), nodes for recall preload, agent, and tools.

**Q: What LLM powers chat?**  
A: **Gemini 2.5 Flash** via Google AI Studio API key or Vertex AI credentials.

**Q: What vector database is used?**  
A: **Pinecone** — two serverless indexes (domain knowledge + session memory), 768 dimensions, cosine metric.

## Trust & citations

**Q: How do I know answers are grounded?**  
A: Expand **References** on any reply that used retrieval. Inline `[1]`, `[2]` markers in the answer match the numbered reference list.

**Q: Can I see what the agent did internally?**  
A: Expand **Agent tools** to see `save_memory`, `recall_memory`, and `retrieve_domain` activity.

## Deployment

**Q: Fastest way to run locally?**  
A: `cp .env.example .env`, add keys, then `./deploy.sh` (Docker) or `streamlit run app/main.py`.

**Q: How do I deploy a public demo?**  
A: Streamlit Community Cloud with main file `app/main.py`. Set `AUTO_INDEX_SAMPLE_CORPUS=true` in secrets for a pre-loaded knowledge base. Full steps: `docs/DEPLOY.md`.

## Demo questions

- How do I get started without uploading files?
- What LLM and vector database does the demo use?
- How are source citations shown in the UI?

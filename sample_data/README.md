# Memory Agent Chat — Sample Corpus

Bundled demo documents for live showcases, Docker deploys, and Streamlit Cloud. Index them in one click from the sidebar or via CLI.

| File | Purpose |
|------|---------|
| `memory_agent_overview.md` | Architecture, features, and tech stack |
| `rag_pipeline_guide.md` | Ingestion, MMR retrieval, and tuning parameters |
| `agent_tools_reference.md` | `save_memory`, `recall_memory`, `retrieve_domain` |
| `demo_faq.md` | FAQ for reviewers and deploy instructions |

## Index the corpus

**In the app:** Sidebar → **Index sample corpus**

**CLI (before first run or for cloud prep):**

```bash
cp .env.example .env   # add API keys
python scripts/index_sample_corpus.py
```

**Docker / Streamlit Cloud:** set `AUTO_INDEX_SAMPLE_CORPUS=true` to index automatically when the knowledge base is empty.

## Suggested demo flow

1. Click **Index sample corpus** (or rely on auto-index)
2. Ask: *"What tech stack does Memory Agent Chat use?"*
3. Ask: *"How does MMR retrieval work?"*
4. Ask: *"Remember that I'm evaluating this for a portfolio demo."* then *"What do you remember about me?"*
5. Expand **References** and **Agent tools** on each reply

## Sample questions

- What agent framework powers this project?
- Which embedding model indexes documents?
- What is the difference between save_memory and retrieve_domain?
- How do I deploy a public demo on Streamlit Cloud?

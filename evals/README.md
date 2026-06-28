# Evaluation Suite

Measures retrieval and answer quality against a fixed corpus and golden Q&A set.

## Metrics

| Metric | Description |
|--------|-------------|
| **recall@5** | Expected source appears in top-5 retrieved chunks |
| **keyword faithfulness** | Expected keywords appear in the answer |

## Run locally

Requires configured API keys (same as the app).

```bash
# Retrieval-only eval (fast, no LLM cost)
python evals/run_eval.py

# Full eval including live LLM answers
python evals/run_eval.py --live-chat --output evals/results.json
```

## CI

Unit tests for metric functions run in GitHub Actions without API keys.
Full eval is intended for local or scheduled runs with secrets configured.

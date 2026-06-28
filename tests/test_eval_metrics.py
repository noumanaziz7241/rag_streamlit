from evals.metrics import keyword_faithfulness, mean_score, recall_at_k


def test_recall_at_k_hit():
    sources = ["docs/guide.md", "docs/other.md"]
    assert recall_at_k(sources, "guide.md", k=2) == 1.0


def test_recall_at_k_miss():
    sources = ["docs/other.md", "docs/another.md"]
    assert recall_at_k(sources, "guide.md", k=2) == 0.0


def test_keyword_faithfulness_partial():
    answer = "The project uses LangGraph for the agent."
    score = keyword_faithfulness(answer, ["LangGraph", "Pinecone"])
    assert score == 0.5


def test_mean_score():
    assert mean_score([1.0, 0.0]) == 0.5
    assert mean_score([]) == 0.0

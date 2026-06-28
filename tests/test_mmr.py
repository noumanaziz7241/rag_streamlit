from memory_agent.vectorstore.domain_index import cosine_similarity, mmr_select


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_mmr_select_prefers_relevant_items():
    query = [1.0, 0.0]
    candidates = [
        {"score": 0.9, "vector": [1.0, 0.0], "id": "a"},
        {"score": 0.85, "vector": [0.99, 0.01], "id": "b"},
        {"score": 0.4, "vector": [0.0, 1.0], "id": "c"},
    ]
    selected = mmr_select(query, candidates, k=2, lambda_mult=0.7)
    ids = [item["id"] for item in selected]
    assert ids[0] == "a"
    assert len(ids) == 2

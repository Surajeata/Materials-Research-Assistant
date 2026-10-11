"""
Tests for hybrid retrieval and cross-encoder reranking.

Run from the project root:
    python -m pytest tests/test_hybrid_reranked.py -v
"""

import pytest

from src import hybrid_retriever


def make_results(ids, documents, distances=None, metadatas=None):
    """Create a ChromaDB-style result dictionary."""
    count = len(ids)

    if distances is None:
        distances = [0.1] * count

    if metadatas is None:
        metadatas = [
            {"source": f"paper_{i}.pdf", "page": i + 1}
            for i in range(count)
        ]

    return {
        "ids": [ids],
        "documents": [documents],
        "metadatas": [metadatas],
        "distances": [distances],
    }


def test_empty_query_returns_empty_results(monkeypatch):
    """An empty query must not trigger semantic retrieval."""

    def unexpected_search(*args, **kwargs):
        pytest.fail("Semantic retrieval should not run for an empty query.")

    monkeypatch.setattr(
        hybrid_retriever,
        "search_collection",
        unexpected_search,
    )

    result = hybrid_retriever.search_hybrid(
        collection=object(),
        query="   ",
    )

    assert result["ids"] == [[]]
    assert result["documents"] == [[]]
    assert result["metadatas"] == [[]]
    assert result["distances"] == [[]]


def test_hybrid_retrieval_combines_semantic_and_lexical_scores(
    monkeypatch,
):
    """Hybrid retrieval should return scores and preserve metadata."""

    documents = [
        "Magnesium alloy deformation improves with grain refinement.",
        "Magnesium alloys exhibit different corrosion behaviour.",
        "Aluminium processing involves extrusion and forging.",
    ]

    ids = ["chunk_1", "chunk_2", "chunk_3"]

    metadatas = [
        {"source": "paper_a.pdf", "page": 2},
        {"source": "paper_b.pdf", "page": 4},
        {"source": "paper_c.pdf", "page": 7},
    ]

    semantic_results = make_results(
        ids=ids,
        documents=documents,
        distances=[0.1, 0.2, 0.8],
        metadatas=metadatas,
    )

    calls = {}

    def fake_search_collection(collection, query, top_k=5):
        calls["collection"] = collection
        calls["query"] = query
        calls["top_k"] = top_k
        return semantic_results

    monkeypatch.setattr(
        hybrid_retriever,
        "search_collection",
        fake_search_collection,
    )

    collection = object()

    result = hybrid_retriever.search_hybrid(
        collection=collection,
        query="magnesium alloy deformation",
        top_k=3,
        candidate_k=10,
        semantic_weight=0.7,
    )

    assert calls["collection"] is collection
    assert calls["query"] == "magnesium alloy deformation"
    assert calls["top_k"] == 10

    assert len(result["ids"][0]) == 3
    assert len(result["documents"][0]) == 3
    assert len(result["metadatas"][0]) == 3
    assert len(result["distances"][0]) == 3
    assert len(result["hybrid_scores"][0]) == 3

    assert result["ids"][0][0] == "chunk_1"
    assert result["metadatas"][0][0] == metadatas[0]

    scores = result["hybrid_scores"][0]

    assert all(isinstance(score, float) for score in scores)
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_hybrid_retrieval_respects_top_k(monkeypatch):
    """Only the requested number of results should be returned."""

    semantic_results = make_results(
        ids=["a", "b", "c", "d"],
        documents=[
            "Magnesium alloy deformation",
            "Magnesium alloy corrosion",
            "Aluminium deformation",
            "Steel heat treatment",
        ],
        distances=[0.1, 0.2, 0.3, 0.4],
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "search_collection",
        lambda *args, **kwargs: semantic_results,
    )

    result = hybrid_retriever.search_hybrid(
        collection=object(),
        query="magnesium alloy deformation",
        top_k=2,
        candidate_k=4,
    )

    assert len(result["ids"][0]) == 2
    assert len(result["documents"][0]) == 2
    assert len(result["hybrid_scores"][0]) == 2


def test_reranked_search_calls_existing_reranker(monkeypatch):
    """Hybrid candidates must be passed to the existing reranker."""

    semantic_results = make_results(
        ids=["chunk_a", "chunk_b", "chunk_c"],
        documents=[
            "Magnesium alloys are used in lightweight structures.",
            "Grain refinement improves magnesium alloy deformation.",
            "Aluminium alloys are used in aerospace applications.",
        ],
        distances=[0.1, 0.2, 0.3],
    )

    calls = {}

    def fake_search_collection(collection, query, top_k=5):
        calls["retrieval_top_k"] = top_k
        return semantic_results

    def fake_rerank_results(query, results, top_k=5):
        calls["reranker_query"] = query
        calls["reranker_results"] = results
        calls["reranker_top_k"] = top_k

        # Simulate a reranker changing the candidate order.
        reordered = {
            "ids": [["chunk_b", "chunk_a"]][:],
            "documents": [[
                "Grain refinement improves magnesium alloy deformation.",
                "Magnesium alloys are used in lightweight structures.",
            ]],
            "metadatas": [[
                results["metadatas"][0][1],
                results["metadatas"][0][0],
            ]],
            "distances": [[0.2, 0.1]],
        }

        return reordered

    monkeypatch.setattr(
        hybrid_retriever,
        "search_collection",
        fake_search_collection,
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "rerank_results",
        fake_rerank_results,
    )

    result = hybrid_retriever.search_hybrid_reranked(
        collection=object(),
        query="magnesium alloy deformation",
        top_k=2,
        candidate_k=3,
    )

    assert calls["retrieval_top_k"] == 3
    assert calls["reranker_query"] == "magnesium alloy deformation"
    assert calls["reranker_top_k"] == 2

    # Verify that the reranker received hybrid scores.
    assert "hybrid_scores" in calls["reranker_results"]

    # Verify that the returned results come from the reranker.
    assert result["ids"][0] == ["chunk_b", "chunk_a"]
    assert len(result["documents"][0]) == 2


def test_reranked_search_handles_no_candidates(monkeypatch):
    """The reranker should not run when retrieval finds nothing."""

    empty_results = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }

    def unexpected_rerank(*args, **kwargs):
        pytest.fail("Reranking should not run without candidates.")

    monkeypatch.setattr(
        hybrid_retriever,
        "search_collection",
        lambda *args, **kwargs: empty_results,
    )

    monkeypatch.setattr(
        hybrid_retriever,
        "rerank_results",
        unexpected_rerank,
    )

    result = hybrid_retriever.search_hybrid_reranked(
        collection=object(),
        query="magnesium alloy deformation",
        top_k=5,
        candidate_k=10,
    )

    assert result["ids"] == [[]]
    assert result["documents"] == [[]]

import re
import numpy as np

from src.retriever import search_collection
from src.reranker import rerank_results


def tokenize(text):
    """Tokenize a query or document for lexical matching."""
    return set(
        re.findall(r"[a-zA-Z0-9]+", text.lower())
    )


def lexical_score(query, document):
    """
    Calculate a simple lexical relevance score.
    Rewards query terms that occur in the document.
    """
    query_terms = tokenize(query)
    document_terms = tokenize(document)

    if not query_terms or not document_terms:
        return 0.0

    return len(query_terms & document_terms) / len(query_terms)


def search_hybrid(
    collection,
    query,
    top_k=5,
    candidate_k=20,
    semantic_weight=0.7,
):
    """
    Combine semantic retrieval with lexical matching.

    Returns the same basic ChromaDB-style result structure
    expected by the existing reranker.
    """

    if not query or not query.strip():
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # Retrieve a wider pool through the existing
    # semantic retrieval pipeline.
    semantic_results = search_collection(
        collection,
        query,
        top_k=max(candidate_k, top_k),
    )

    documents = semantic_results.get(
        "documents", [[]]
    )[0]

    metadatas = semantic_results.get(
        "metadatas", [[]]
    )[0]

    ids = semantic_results.get(
        "ids", [[]]
    )[0]

    distances = semantic_results.get(
        "distances", [[]]
    )[0]

    if not documents:
        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

    # Convert semantic distances to a comparable
    # similarity score. Smaller distance is better.
    semantic_scores = [
        1.0 / (1.0 + max(float(d), 0.0))
        for d in distances
    ]

    max_semantic = max(semantic_scores) or 1.0

    semantic_scores = [
        score / max_semantic
        for score in semantic_scores
    ]

    combined = []

    for i, document in enumerate(documents):
        lexical = lexical_score(query, document)
        semantic = semantic_scores[i]

        score = (
            semantic_weight * semantic
            + (1.0 - semantic_weight) * lexical
        )

        combined.append({
            "id": ids[i],
            "document": document,
            "metadata": metadatas[i],
            "distance": float(distances[i]),
            "hybrid_score": score,
        })

    combined.sort(
        key=lambda item: item["hybrid_score"],
        reverse=True,
    )

    selected = combined[:top_k]

    return {
        "ids": [[item["id"] for item in selected]],
        "documents": [
            [item["document"] for item in selected]
        ],
        "metadatas": [
            [item["metadata"] for item in selected]
        ],
        "distances": [
            [item["distance"] for item in selected]
        ],
        "hybrid_scores": [
            [item["hybrid_score"] for item in selected]
        ],
    }


def search_hybrid_reranked(
    collection,
    query,
    top_k=5,
    candidate_k=20,
):
    """
    Run hybrid retrieval followed by the existing
    cross-encoder reranker.
    """

    candidates = search_hybrid(
        collection=collection,
        query=query,
        top_k=max(candidate_k, top_k),
        candidate_k=candidate_k,
    )

    if not candidates["documents"][0]:
        return candidates

    reranked = rerank_results(
        query=query,
        results=candidates,
        top_k=top_k,
    )

    return reranked

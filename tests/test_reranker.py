import sys
from pathlib import Path


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)


# ============================================================
# IMPORT PROJECT MODULES
# ============================================================

from src.vector_store import create_collection
from src.retriever import search_collection
from src.reranker import rerank_results


# ============================================================
# CREATE COLLECTION
# ============================================================

collection = create_collection()


# ============================================================
# TEST QUERIES
# ============================================================

queries = [

    "What processing strategies improve the plastic "
    "deformability of magnesium alloys?",

    "How do rare-earth elements affect texture "
    "evolution during processing of magnesium alloys?",

    "How does grain refinement affect the plastic "
    "deformability of magnesium alloys?"
]


# ============================================================
# TEST HEADER
# ============================================================

print()

print(
    "=" * 90
)

print(
    "CROSS-ENCODER RERANKER TEST"
)

print(
    "=" * 90
)


# ============================================================
# RUN TESTS
# ============================================================

for query in queries:

    print()

    print(
        "=" * 90
    )

    print(
        f"QUERY:\n{query}"
    )

    print(
        "=" * 90
    )


    # ========================================================
    # INITIAL RETRIEVAL
    #
    # Retrieve a larger candidate pool first.
    # ========================================================

    results = search_collection(
        collection,
        query,
        top_k=20
    )


    documents = results[
        "documents"
    ][0]


    print()

    print(
        "Initial candidates: "
        f"{len(documents)}"
    )


    # ========================================================
    # RERANK
    # ========================================================

    reranked = rerank_results(
        query,
        results,
        top_k=5
    )


    reranked_documents = (
        reranked["documents"][0]
    )

    reranked_metadatas = (
        reranked["metadatas"][0]
    )

    rerank_scores = (
        reranked["rerank_scores"][0]
    )


    print()

    print(
        "Final reranked results: "
        f"{len(reranked_documents)}"
    )


    # ========================================================
    # DISPLAY RERANKED RESULTS
    # ========================================================

    for i in range(
        len(reranked_documents)
    ):

        metadata = (
            reranked_metadatas[i]
        )


        source = metadata.get(
            "source",
            "Unknown"
        )


        page = metadata.get(
            "page_number",
            "Unknown"
        )


        print()

        print(
            f"RERANKED RESULT {i + 1}"
        )


        print(
            f"Source: {source}"
        )


        print(
            f"Page: {page}"
        )


        print(
            f"Rerank score: "
            f"{rerank_scores[i]:.4f}"
        )


        print(
            "Text:"
        )


        print(
            reranked_documents[i][:800]
        )


        print(
            "-" * 90
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print(
    "=" * 90
)

print(
    "RERANKER TEST COMPLETE"
)

print(
    "=" * 90
)
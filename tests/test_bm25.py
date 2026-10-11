import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

from src.vector_store import create_collection
from src.retriever import search_bm25


collection = create_collection()


query = """
magnesium alloy rare earth texture
plastic deformability
"""


results = search_bm25(
    collection,
    query,
    top_k=5
)


print()
print("=" * 70)
print("BM25 SEARCH TEST")
print("=" * 70)


print(
    f"\nRetrieved results: {len(results)}\n"
)


for i, result in enumerate(
    results,
    start=1
):

    metadata = result[
        "metadata"
    ]

    print(
        f"\nRESULT {i}"
    )

    print(
        f"Source: "
        f"{metadata.get('source')}"
    )

    print(
        f"Page: "
        f"{metadata.get('page_number')}"
    )

    print(
        f"BM25 Score: "
        f"{result['bm25_score']:.4f}"
    )

    print(
        "Text:"
    )

    print(
        result["document"][:500]
    )

    print(
        "-" * 70
    )
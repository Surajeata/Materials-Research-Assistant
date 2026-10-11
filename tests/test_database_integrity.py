import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

from src.vector_store import create_collection


collection = create_collection()


print()
print("=" * 80)
print("CHROMADB INTEGRITY TEST")
print("=" * 80)


print()
print(
    f"TOTAL CHUNKS: {collection.count()}"
)


# ---------------------------------------------------------
# Get all stored metadata
# ---------------------------------------------------------

results = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)


documents = results[
    "documents"
]

metadatas = results[
    "metadatas"
]


# ---------------------------------------------------------
# Count information by document
# ---------------------------------------------------------

document_stats = {}


for metadata in metadatas:

    source = metadata.get(
        "source",
        "UNKNOWN"
    )

    is_reference = metadata.get(
        "is_reference",
        False
    )


    if source not in document_stats:

        document_stats[source] = {
            "total": 0,
            "reference": 0,
            "non_reference": 0
        }


    document_stats[source][
        "total"
    ] += 1


    if is_reference:

        document_stats[source][
            "reference"
        ] += 1

    else:

        document_stats[source][
            "non_reference"
        ] += 1


# ---------------------------------------------------------
# Print statistics
# ---------------------------------------------------------

print()
print("=" * 80)
print("DOCUMENT STATISTICS")
print("=" * 80)


for source, stats in document_stats.items():

    print()

    print(
        f"Document: {source}"
    )

    print(
        f"Total chunks: "
        f"{stats['total']}"
    )

    print(
        f"Reference chunks: "
        f"{stats['reference']}"
    )

    print(
        f"Non-reference chunks: "
        f"{stats['non_reference']}"
    )


# ---------------------------------------------------------
# Check for missing metadata
# ---------------------------------------------------------

print()
print("=" * 80)
print("METADATA VALIDATION")
print("=" * 80)


required_fields = [
    "source",
    "page_number",
    "document_id",
    "is_reference"
]


missing_metadata = 0


for index, metadata in enumerate(
    metadatas
):

    missing = [
        field
        for field in required_fields
        if field not in metadata
    ]


    if missing:

        missing_metadata += 1

        print(
            f"Chunk {index} "
            f"missing: {missing}"
        )


if missing_metadata == 0:

    print(
        "✓ All chunks contain required metadata."
    )

else:

    print(
        f"✗ {missing_metadata} "
        f"chunks have missing metadata."
    )


# ---------------------------------------------------------
# Check page numbers
# ---------------------------------------------------------

print()
print("=" * 80)
print("PAGE NUMBER VALIDATION")
print("=" * 80)


invalid_pages = 0


for metadata in metadatas:

    page = metadata.get(
        "page_number"
    )


    if not isinstance(
        page,
        int
    ) or page <= 0:

        invalid_pages += 1


if invalid_pages == 0:

    print(
        "✓ All chunks have valid page numbers."
    )

else:

    print(
        f"✗ {invalid_pages} "
        f"chunks have invalid page numbers."
    )


# ---------------------------------------------------------
# Final result
# ---------------------------------------------------------

print()
print("=" * 80)
print("DATABASE TEST COMPLETE")
print("=" * 80)

if (
    missing_metadata == 0
    and invalid_pages == 0
):

    print(
        "✓ DATABASE IS HEALTHY"
    )

else:

    print(
        "✗ DATABASE NEEDS ATTENTION"
    )
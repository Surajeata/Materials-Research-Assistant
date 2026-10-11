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


# ============================================================
# CREATE CHROMA COLLECTION
# ============================================================

collection = create_collection()


# ============================================================
# DIVERSITY FILTER
# ============================================================

def select_diverse_results(
    results,
    top_k=5
):
    """
    Select diverse retrieval results.

    Selection priority:

    1. Relevant results
    2. Unique source + page
    3. Avoid duplicate text
    4. Prefer evidence from different pages

    The function first selects results from unique pages.

    Only when there are genuinely not enough unique pages
    available will repeated pages be allowed.
    """

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]


    selected_documents = []
    selected_metadatas = []
    selected_distances = []


    seen_pages = set()
    seen_text = set()


    # ========================================================
    # PASS 1
    #
    # Select one result from each unique page.
    # ========================================================

    for i in range(
        len(documents)
    ):

        metadata = metadatas[i]


        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            "Unknown"
        )


        page_key = (
            source,
            page
        )


        # ----------------------------------------------------
        # Normalize text for duplicate detection
        # ----------------------------------------------------

        normalized_text = (
            documents[i]
            .lower()
            .strip()
        )


        # Use the first 300 characters
        # as a lightweight duplicate signature.

        text_key = normalized_text[:300]


        # ----------------------------------------------------
        # Skip duplicate text
        # ----------------------------------------------------

        if text_key in seen_text:
            continue


        # ----------------------------------------------------
        # Skip duplicate pages
        # ----------------------------------------------------

        if page_key in seen_pages:
            continue


        # ----------------------------------------------------
        # Select result
        # ----------------------------------------------------

        seen_pages.add(
            page_key
        )

        seen_text.add(
            text_key
        )


        selected_documents.append(
            documents[i]
        )

        selected_metadatas.append(
            metadata
        )

        selected_distances.append(
            distances[i]
        )


        # Stop when enough results
        # have been selected.

        if len(
            selected_documents
        ) >= top_k:

            break


    # ========================================================
    # PASS 2
    #
    # Continue searching for unique pages.
    #
    # This provides an additional safety layer in case
    # the first pass did not fill all available slots.
    # ========================================================

    if len(
        selected_documents
    ) < top_k:

        for i in range(
            len(documents)
        ):

            metadata = metadatas[i]


            source = metadata.get(
                "source",
                "Unknown"
            )

            page = metadata.get(
                "page_number",
                "Unknown"
            )


            page_key = (
                source,
                page
            )


            # ------------------------------------------------
            # Do not allow another chunk from an already
            # selected page.
            # ------------------------------------------------

            if page_key in seen_pages:
                continue


            normalized_text = (
                documents[i]
                .lower()
                .strip()
            )


            text_key = normalized_text[:300]


            if text_key in seen_text:
                continue


            seen_pages.add(
                page_key
            )

            seen_text.add(
                text_key
            )


            selected_documents.append(
                documents[i]
            )

            selected_metadatas.append(
                metadata
            )

            selected_distances.append(
                distances[i]
            )


            if len(
                selected_documents
            ) >= top_k:

                break


    # ========================================================
    # PASS 3
    #
    # Only allow duplicate pages if there are genuinely
    # fewer than top_k unique pages available.
    # ========================================================

    if len(
        selected_documents
    ) < top_k:

        for i in range(
            len(documents)
        ):

            # Avoid selecting the exact same chunk twice.

            if documents[i] in selected_documents:
                continue


            selected_documents.append(
                documents[i]
            )

            selected_metadatas.append(
                metadatas[i]
            )

            selected_distances.append(
                distances[i]
            )


            if len(
                selected_documents
            ) >= top_k:

                break


    # ========================================================
    # RETURN FILTERED RESULTS
    # ========================================================

    return {
        "documents": [
            selected_documents
        ],

        "metadatas": [
            selected_metadatas
        ],

        "distances": [
            selected_distances
        ]
    }


# ============================================================
# TEST QUERIES
# ============================================================

queries = [

    "What processing strategies improve the plastic "
    "deformability of magnesium alloys?",

    "How do rare-earth elements affect texture "
    "evolution during processing of magnesium alloys?",

    "What alloying strategies are used to improve "
    "magnesium alloy formability?",

    "What are the main deformation mechanisms "
    "in magnesium alloys?",

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
    "FINAL RETRIEVAL PIPELINE TEST"
)

print(
    "=" * 90
)


# ============================================================
# RUN EACH QUERY
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
    # RETRIEVE A LARGER CANDIDATE POOL
    #
    # We intentionally retrieve 20 candidates instead of 10.
    #
    # This gives the diversity filter enough candidates to
    # find results from different pages.
    # ========================================================

    raw_results = search_collection(
        collection,
        query,
        top_k=20
    )


    # ========================================================
    # EXTRACT RAW RESULTS
    # ========================================================

    raw_documents = raw_results[
        "documents"
    ][0]

    raw_metadatas = raw_results[
        "metadatas"
    ][0]

    raw_distances = raw_results[
        "distances"
    ][0]


    print()

    print(
        "Candidates before diversity filtering: "
        f"{len(raw_documents)}"
    )


    # ========================================================
    # APPLY DIVERSITY FILTER
    # ========================================================

    results = select_diverse_results(
        raw_results,
        top_k=5
    )


    # ========================================================
    # EXTRACT FINAL RESULTS
    # ========================================================

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]


    print()

    print(
        "Final results after diversity filtering: "
        f"{len(documents)}"
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    for i in range(
        len(documents)
    ):

        metadata = metadatas[i]


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
            f"RESULT {i + 1}"
        )


        print(
            f"Source: {source}"
        )


        print(
            f"Page: {page}"
        )


        print(
            f"Distance: "
            f"{distances[i]:.4f}"
        )


        print(
            "Text:"
        )


        print(
            documents[i][:800]
        )


        print(
            "-" * 90
        )


    # ========================================================
    # SHOW UNIQUE PAGES
    # ========================================================

    unique_pages = []


    for metadata in metadatas:

        source = metadata.get(
            "source",
            "Unknown"
        )

        page = metadata.get(
            "page_number",
            "Unknown"
        )


        page_label = (
            f"{source} "
            f"-> Page {page}"
        )


        if page_label not in unique_pages:

            unique_pages.append(
                page_label
            )


    print()

    print(
        "UNIQUE PAGES:"
    )


    for page in unique_pages:

        print(
            f"    {page}"
        )


    print()

    print(
        "=" * 90
    )


# ============================================================
# FINAL MESSAGE
# ============================================================

print()

print(
    "=" * 90
)

print(
    "RETRIEVAL TEST COMPLETE"
)

print(
    "=" * 90
)
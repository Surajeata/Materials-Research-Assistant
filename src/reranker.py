import re

from sentence_transformers import CrossEncoder


# ============================================================
# RERANKER CONFIGURATION
# ============================================================

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

model = CrossEncoder(
    MODEL_NAME
)


# ============================================================
# GENERIC / LOW-VALUE CONTENT DETECTION
# ============================================================

GENERIC_TERMS = [
    "received",
    "revised",
    "accepted",
    "published online",
    "available online",
    "department of",
    "university",
    "corresponding author",
    "email address",
    "copyright",
    "open access",
    "creativecommons",
    "doi:",
    "issn",
    "hosted by",
    "science direct",
    "springer",
]


def generic_content_penalty(text):
    """
    Detect chunks that contain mostly publication metadata,
    author information, or other low-value introductory content.

    These chunks can sometimes receive a high CrossEncoder
    score because they contain important words from the paper,
    even though they do not provide useful scientific evidence.
    """

    if not text:
        return 0.0

    text_lower = text.lower()

    penalty = 0.0

    # --------------------------------------------------------
    # Publication / author metadata
    # --------------------------------------------------------

    metadata_hits = sum(
        1
        for term in GENERIC_TERMS
        if term in text_lower
    )

    if metadata_hits >= 2:
        penalty += 1.5

    elif metadata_hits == 1:
        penalty += 0.5


    # --------------------------------------------------------
    # Strong title / abstract signal
    # --------------------------------------------------------

    first_part = text_lower[:700]

    if (
        "review" in first_part
        and (
            "abstract" in first_part
            or "received" in first_part
        )
    ):
        penalty += 1.0


    # --------------------------------------------------------
    # Author-heavy chunk
    # --------------------------------------------------------

    if (
        "university" in text_lower
        and "department" in text_lower
        and "received" in text_lower
    ):
        penalty += 2.0


    return penalty


# ============================================================
# DOMAIN RELEVANCE BOOST
# ============================================================

DOMAIN_KEYWORDS = [
    "magnesium",
    "mg alloy",
    "mg alloys",
    "rare-earth",
    "rare earth",
    "texture",
    "texture evolution",
    "texture weakening",
    "basal texture",
    "non-basal slip",
    "slip",
    "twinning",
    "deformation",
    "plastic deformability",
    "formability",
    "ductility",
    "grain refinement",
    "grain size",
    "recrystallization",
    "dynamic recrystallization",
    "microstructure",
    "processing",
    "rolling",
    "extrusion",
    "forging",
    "alloying",
    "c/a ratio",
    "crss",
]


def domain_relevance_score(
    query,
    text
):
    """
    Calculate a lightweight domain relevance score.

    This does not replace the CrossEncoder.

    It acts only as a small adjustment so that chunks
    containing important materials-science concepts are
    preferred over generic paper metadata.
    """

    if not text:
        return 0.0

    query_lower = query.lower()
    text_lower = text.lower()

    score = 0.0


    # --------------------------------------------------------
    # Query-specific keyword overlap
    # --------------------------------------------------------

    query_terms = re.findall(
        r"\b[a-zA-Z][a-zA-Z\-]{2,}\b",
        query_lower
    )

    unique_query_terms = set(
        query_terms
    )

    for term in unique_query_terms:

        if term in text_lower:

            score += 0.10


    # --------------------------------------------------------
    # Materials-science terminology
    # --------------------------------------------------------

    for keyword in DOMAIN_KEYWORDS:

        if keyword in text_lower:

            score += 0.08


    # --------------------------------------------------------
    # Strong scientific relationship signals
    # --------------------------------------------------------

    strong_phrases = [
        "affect texture",
        "texture weakening",
        "modify the texture",
        "texture evolution",
        "promoting non-basal slip",
        "reducing texture intensity",
        "improving ductility",
        "improve formability",
        "plastic deformability",
        "deformation mechanisms",
        "grain refinement",
        "dynamic recrystallization",
        "critical resolved shear stress",
        "crss differences",
    ]

    for phrase in strong_phrases:

        if phrase in text_lower:

            score += 0.25


    return score


# ============================================================
# PAGE / CONTENT DIVERSITY
# ============================================================

def apply_diversity_filter(
    candidates,
    top_k=5
):
    """
    Select final results while avoiding excessive
    duplication from the same page.

    The first result from a page is retained, and
    additional chunks from the same page are skipped
    until enough unique pages have been selected.
    """

    selected = []

    seen_pages = set()


    # --------------------------------------------------------
    # First pass: unique pages
    # --------------------------------------------------------

    for candidate in candidates:

        metadata = candidate.get(
            "metadata",
            {}
        )

        page = metadata.get(
            "page_number"
        )

        source = metadata.get(
            "source"
        )

        page_key = (
            source,
            page
        )

        if page_key in seen_pages:
            continue

        seen_pages.add(
            page_key
        )

        selected.append(
            candidate
        )

        if len(selected) >= top_k:
            break


    # --------------------------------------------------------
    # Second pass: fill remaining slots if necessary
    # --------------------------------------------------------

    if len(selected) < top_k:

        selected_ids = {
            candidate.get("id")
            for candidate in selected
        }

        for candidate in candidates:

            candidate_id = candidate.get(
                "id"
            )

            if candidate_id in selected_ids:
                continue

            selected.append(
                candidate
            )

            if len(selected) >= top_k:
                break


    return selected


# ============================================================
# MAIN RERANKING FUNCTION
# ============================================================

def rerank_results(
    query,
    results,
    top_k=5
):
    """
    Rerank retrieved ChromaDB results using a CrossEncoder.

    Pipeline:

        semantic retrieval
                ↓
        CrossEncoder scoring
                ↓
        domain relevance adjustment
                ↓
        generic-content penalty
                ↓
        diversity filtering
                ↓
        final top-k results
    """

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    ids = results.get(
        "ids",
        [[]]
    )[0]

    distances = results.get(
        "distances",
        [[]]
    )[0]


    # --------------------------------------------------------
    # Empty-result protection
    # --------------------------------------------------------

    if not documents:

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "rerank_scores": [[]]
        }


    # --------------------------------------------------------
    # Build query-document pairs
    # --------------------------------------------------------

    pairs = []

    for document in documents:

        pairs.append(
            [
                query,
                document
            ]
        )


    # --------------------------------------------------------
    # CrossEncoder scores
    # --------------------------------------------------------

    cross_encoder_scores = model.predict(
        pairs
    )


    # --------------------------------------------------------
    # Build candidate objects
    # --------------------------------------------------------

    candidates = []


    for i in range(
        len(documents)
    ):

        metadata = (
            metadatas[i]
            if i < len(metadatas)
            else {}
        )

        document_id = (
            ids[i]
            if i < len(ids)
            else f"candidate_{i}"
        )


        cross_score = float(
            cross_encoder_scores[i]
        )


        # ----------------------------------------------------
        # Domain relevance adjustment
        # ----------------------------------------------------

        domain_score = (
            domain_relevance_score(
                query,
                documents[i]
            )
        )


        # ----------------------------------------------------
        # Generic-content penalty
        # ----------------------------------------------------

        generic_penalty = (
            generic_content_penalty(
                documents[i]
            )
        )


        # ----------------------------------------------------
        # Final score
        # ----------------------------------------------------

        final_score = (
            cross_score
            + domain_score
            - generic_penalty
        )


        candidates.append({

            "id": document_id,

            "document": documents[i],

            "metadata": metadata,

            "original_distance": (
                distances[i]
                if i < len(distances)
                else None
            ),

            "cross_encoder_score": (
                cross_score
            ),

            "domain_score": (
                domain_score
            ),

            "generic_penalty": (
                generic_penalty
            ),

            "rerank_score": (
                final_score
            )

        })


    # --------------------------------------------------------
    # Sort by final score
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["rerank_score"],
        reverse=True
    )


    print()
    print(
        f"Initial candidates: "
        f"{len(candidates)}"
    )


    # --------------------------------------------------------
    # Diversity filtering
    # --------------------------------------------------------

    selected = apply_diversity_filter(
        candidates,
        top_k=top_k
    )


    print()
    print(
        f"Final reranked results: "
        f"{len(selected)}"
    )


    # --------------------------------------------------------
    # Convert back to Chroma-style result format
    # --------------------------------------------------------

    return {
        "ids": [[
            candidate["id"]
            for candidate in selected
        ]],

        "documents": [[
            candidate["document"]
            for candidate in selected
        ]],

        "metadatas": [[
            candidate["metadata"]
            for candidate in selected
        ]],

        "distances": [[
            candidate["original_distance"]
            for candidate in selected
        ]],

        "rerank_scores": [[
            candidate["rerank_score"]
            for candidate in selected
        ]],

        "cross_encoder_scores": [[
            candidate["cross_encoder_score"]
            for candidate in selected
        ]],

        "domain_scores": [[
            candidate["domain_score"]
            for candidate in selected
        ]],

        "generic_penalties": [[
            candidate["generic_penalty"]
            for candidate in selected
        ]]
    }
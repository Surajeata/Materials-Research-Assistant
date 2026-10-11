from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)
from rank_bm25 import BM25Okapi
import numpy as np
import re


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

reranker = CrossEncoder(
    RERANKER_MODEL_NAME
)


def looks_like_reference_chunk(text):
    """
    Detect chunks that mainly belong to a reference/bibliography section.

    This is an additional safety check. The primary reference
    filtering is done using the 'is_reference' metadata.
    """

    text_lower = text.lower().strip()

    reference_headings = [
        "references",
        "bibliography",
        "literature cited",
        "references and notes",
        "reference list"
    ]

    beginning = text_lower[:200]

    for heading in reference_headings:

        if re.search(
            rf"\b{re.escape(heading)}\b",
            beginning
        ):
            return True

    return False


def search_collection(collection, query, top_k=5):
    """
    Retrieve relevant research-paper chunks.

    Steps:
    1. Create query embedding.
    2. Search a larger candidate pool.
    3. Remove reference-section chunks.
    4. Calculate cosine similarity.
    5. Apply an absolute relevance threshold.
    6. Reduce duplicate pages.
    7. Return the strongest relevant chunks.
    """

    # ---------------------------------------------------------
    # 1. Create query embedding
    # ---------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]


    # ---------------------------------------------------------
    # 2. Check whether the collection contains documents
    # ---------------------------------------------------------

    total_chunks = collection.count()

    if total_chunks == 0:

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }


    # ---------------------------------------------------------
    # 3. Search a larger candidate pool
    # ---------------------------------------------------------

    candidate_count = min(
        max(top_k * 4, 20),
        total_chunks
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=candidate_count,
        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )


    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]
    embeddings = results["embeddings"][0]


    # ---------------------------------------------------------
    # 4. Calculate similarity and filter references
    # ---------------------------------------------------------

    candidates = []

    for i in range(len(documents)):

        text = documents[i]
        metadata = metadatas[i]


        # -----------------------------------------------------
        # IMPORTANT:
        # Never retrieve chunks explicitly marked as belonging
        # to the reference/bibliography section.
        # -----------------------------------------------------

        if metadata.get(
            "is_reference",
            False
        ):
            continue


        # -----------------------------------------------------
        # Additional text-based reference check
        # -----------------------------------------------------

        if looks_like_reference_chunk(text):
            continue


        # -----------------------------------------------------
        # Calculate cosine similarity
        # -----------------------------------------------------

        embedding = np.array(
            embeddings[i]
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            continue

        embedding = embedding / norm

        similarity = float(
            np.dot(
                query_embedding,
                embedding
            )
        )


        candidates.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata,
            "similarity": similarity
        })


    # ---------------------------------------------------------
    # 5. Sort by similarity
    # ---------------------------------------------------------

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )


    # ---------------------------------------------------------
    # 6. Apply minimum relevance threshold
    # ---------------------------------------------------------

    MIN_SIMILARITY = 0.45

    relevant_candidates = [
        candidate
        for candidate in candidates
        if candidate["similarity"] >= MIN_SIMILARITY
    ]


    # ---------------------------------------------------------
    # 7. If nothing is sufficiently relevant
    # ---------------------------------------------------------

    if not relevant_candidates:

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }


    # ---------------------------------------------------------
    # 8. Reduce duplicate pages
    # ---------------------------------------------------------

    selected = []

    seen_pages = set()


    for candidate in relevant_candidates:

        metadata = candidate["metadata"]

        document_id = metadata["document_id"]
        page_number = metadata["page_number"]

        page_key = (
            document_id,
            page_number
        )


        # Don't return multiple chunks from the
        # same page when other relevant pages exist.

        if page_key in seen_pages:
            continue


        seen_pages.add(page_key)

        selected.append(candidate)


        if len(selected) >= top_k:
            break


    # ---------------------------------------------------------
    # 9. Fill remaining slots if necessary
    # ---------------------------------------------------------

    if len(selected) < min(
        top_k,
        len(relevant_candidates)
    ):

        selected_ids = {
            candidate["id"]
            for candidate in selected
        }


        for candidate in relevant_candidates:

            if candidate["id"] in selected_ids:
                continue


            selected.append(candidate)


            if len(selected) >= top_k:
                break


    # ---------------------------------------------------------
    # 10. Prepare final results
    # ---------------------------------------------------------

    final_ids = [
        candidate["id"]
        for candidate in selected
    ]

    final_documents = [
        candidate["document"]
        for candidate in selected
    ]

    final_metadatas = [
        candidate["metadata"]
        for candidate in selected
    ]

    final_distances = [
        1 - candidate["similarity"]
        for candidate in selected
    ]


    return {
        "ids": [final_ids],
        "documents": [final_documents],
        "metadatas": [final_metadatas],
        "distances": [final_distances]
    }

def search_collection_for_document(
    collection,
    query,
    source_name,
    top_k=20
):
    """
    Retrieve the most relevant chunks from one
    specific research paper.

    A larger candidate pool is retrieved first so
    that reference filtering and page deduplication
    do not reduce the final result set too aggressively.
    """

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    total_chunks = collection.count()

    if total_chunks == 0:

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # --------------------------------------------------------
    # Retrieve a larger candidate pool
    #
    # Example:
    # top_k = 5
    # candidate_count = 25
    #
    # This gives us enough candidates after filtering.
    # --------------------------------------------------------

    candidate_count = min(
        max(top_k * 5, 25),
        total_chunks
    )

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=candidate_count,

        where={
            "source": source_name
        },

        include=[
            "documents",
            "metadatas",
            "embeddings"
        ]
    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    ids = results[
        "ids"
    ][0]

    embeddings = results[
        "embeddings"
    ][0]

    # --------------------------------------------------------
    # Calculate similarity and filter references
    # --------------------------------------------------------

    candidates = []

    for i in range(
        len(documents)
    ):

        metadata = metadatas[i]

        # ----------------------------------------------------
        # Never retrieve explicitly marked references
        # ----------------------------------------------------

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        # ----------------------------------------------------
        # Additional reference safety check
        # ----------------------------------------------------

        if looks_like_reference_chunk(
            text
        ):
            continue

        # ----------------------------------------------------
        # Calculate cosine similarity
        # ----------------------------------------------------

        embedding = np.array(
            embeddings[i]
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            continue

        embedding = (
            embedding / norm
        )

        similarity = float(
            np.dot(
                query_embedding,
                embedding
            )
        )

        candidates.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata,
            "similarity": similarity
        })

    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    candidates.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Select chunks while avoiding duplicate pages
    # --------------------------------------------------------

    selected = []

    seen_pages = set()

    for candidate in candidates:

        metadata = candidate[
            "metadata"
        ]

        page_number = metadata[
            "page_number"
        ]

        page_key = (
            source_name,
            page_number
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
    # If page deduplication left too few results,
    # fill remaining slots with additional chunks.
    # --------------------------------------------------------

    if len(selected) < min(
        top_k,
        len(candidates)
    ):

        selected_ids = {
            candidate["id"]
            for candidate in selected
        }

        for candidate in candidates:

            if candidate["id"] in selected_ids:
                continue

            selected.append(
                candidate
            )

            if len(selected) >= top_k:
                break

    # --------------------------------------------------------
    # Return Chroma-like structure
    # --------------------------------------------------------

    return {
        "ids": [[
            item["id"]
            for item in selected
        ]],

        "documents": [[
            item["document"]
            for item in selected
        ]],

        "metadatas": [[
            item["metadata"]
            for item in selected
        ]],

        "distances": [[
            1 - item["similarity"]
            for item in selected
        ]]
    }
    """
    Retrieve the most relevant chunks from one
    specific research paper.
    """

    # --------------------------------------------------------
    # Create query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    total_chunks = collection.count()

    if total_chunks == 0:

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }

    # --------------------------------------------------------
    # Search only within selected document
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],

        n_results=min(
            top_k,
            total_chunks
        ),

        where={
            "source": source_name
        },

        include=[
            "documents",
            "metadatas",
            "embeddings",
            "distances"
        ]
    )

    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    ids = results[
        "ids"
    ][0]

    embeddings = results.get(
        "embeddings",
        [[]]
    )[0]

    # --------------------------------------------------------
    # Filter references
    # --------------------------------------------------------

    filtered = []

    for i in range(len(documents)):

        metadata = metadatas[i]

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        if looks_like_reference_chunk(text):
            continue

        # ----------------------------------------------------
        # Calculate cosine similarity
        # ----------------------------------------------------

        embedding = np.array(
            embeddings[i]
        )

        norm = np.linalg.norm(
            embedding
        )

        if norm == 0:
            continue

        embedding = embedding / norm

        similarity = float(
            np.dot(
                query_embedding,
                embedding
            )
        )

        filtered.append({
            "id": ids[i],
            "document": text,
            "metadata": metadata,
            "similarity": similarity
        })

    # --------------------------------------------------------
    # Sort by relevance
    # --------------------------------------------------------

    filtered.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    # --------------------------------------------------------
    # Avoid too many chunks from the same page
    # --------------------------------------------------------

    selected = []

    seen_pages = set()

    for candidate in filtered:

        metadata = candidate[
            "metadata"
        ]

        page_number = metadata[
            "page_number"
        ]

        if page_number in seen_pages:
            continue

        seen_pages.add(
            page_number
        )

        selected.append(
            candidate
        )

        if len(selected) >= top_k:
            break

    # --------------------------------------------------------
    # Return Chroma-like structure
    # --------------------------------------------------------

    return {
        "ids": [[
            item["id"]
            for item in selected
        ]],

        "documents": [[
            item["document"]
            for item in selected
        ]],

        "metadatas": [[
            item["metadata"]
            for item in selected
        ]],

        "distances": [[
            1 - item["similarity"]
            for item in selected
        ]]
    }

def build_bm25_index(
    collection,
    source_name=None
):
    """
    Build a BM25 index from the research-paper
    chunks stored in ChromaDB.

    BM25 is useful for exact keyword and technical
    terminology matching.
    """

    # ---------------------------------------------------------
    # Retrieve documents
    # ---------------------------------------------------------

    if source_name:

        results = collection.get(
            where={
                "source": source_name
            },
            include=[
                "documents",
                "metadatas"
            ]
        )

    else:

        results = collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )


    documents = results.get(
        "documents",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    ids = results.get(
        "ids",
        []
    )


    # ---------------------------------------------------------
    # Remove reference chunks
    # ---------------------------------------------------------

    filtered_documents = []
    filtered_metadatas = []
    filtered_ids = []

    for i in range(
        len(documents)
    ):

        metadata = metadatas[i]

        if metadata.get(
            "is_reference",
            False
        ):
            continue

        text = documents[i]

        if looks_like_reference_chunk(
            text
        ):
            continue

        filtered_documents.append(
            text
        )

        filtered_metadatas.append(
            metadata
        )

        filtered_ids.append(
            ids[i]
        )


    # ---------------------------------------------------------
    # Tokenize documents
    # ---------------------------------------------------------

    tokenized_documents = [
        document.lower().split()
        for document in filtered_documents
    ]


    # ---------------------------------------------------------
    # Create BM25 index
    # ---------------------------------------------------------

    if not tokenized_documents:

        return {
            "bm25": None,
            "documents": [],
            "metadatas": [],
            "ids": []
        }


    bm25 = BM25Okapi(
        tokenized_documents
    )


    return {
        "bm25": bm25,
        "documents": filtered_documents,
        "metadatas": filtered_metadatas,
        "ids": filtered_ids
    }


def search_bm25(
    collection,
    query,
    source_name=None,
    top_k=5
):
    """
    Perform BM25 keyword-based retrieval.

    Unlike semantic retrieval, BM25 is particularly
    useful when the query contains exact technical
    terminology, alloy names, compositions, or
    scientific keywords.
    """

    index = build_bm25_index(
        collection,
        source_name=source_name
    )

    bm25 = index[
        "bm25"
    ]

    documents = index[
        "documents"
    ]

    metadatas = index[
        "metadatas"
    ]

    ids = index[
        "ids"
    ]


    if bm25 is None:

        return []


    # ---------------------------------------------------------
    # Tokenize query
    # ---------------------------------------------------------

    query_tokens = (
        query
        .lower()
        .split()
    )


    # ---------------------------------------------------------
    # Calculate BM25 scores
    # ---------------------------------------------------------

    scores = bm25.get_scores(
        query_tokens
    )


    # ---------------------------------------------------------
    # Sort by BM25 score
    # ---------------------------------------------------------

    ranked_indices = np.argsort(
        scores
    )[::-1]


    results = []


    for index_position in ranked_indices:

        score = float(
            scores[index_position]
        )


        # Ignore documents with no keyword match
        if score <= 0:
            continue


        results.append({

            "id": ids[
                index_position
            ],

            "document": documents[
                index_position
            ],

            "metadata": metadatas[
                index_position
            ],

            "bm25_score": score

        })


        if len(results) >= top_k:
            break


    return results


def search_hybrid(
    collection,
    query,
    source_name=None,
    top_k=5,
    semantic_weight=0.6,
    bm25_weight=0.4
):
    """
    Combine semantic retrieval and BM25 lexical retrieval.

    semantic_weight:
        Importance of semantic similarity.

    bm25_weight:
        Importance of exact keyword matching.
    """

    # ---------------------------------------------------------
    # 1. Semantic retrieval
    # ---------------------------------------------------------

    if source_name:

        semantic_results = search_collection_for_document(
            collection=collection,
            query=query,
            source_name=source_name,
            top_k=max(top_k * 4, 20)
        )

    else:

        semantic_results = search_collection(
            collection=collection,
            query=query,
            top_k=max(top_k * 4, 20)
        )


    # ---------------------------------------------------------
    # 2. BM25 retrieval
    # ---------------------------------------------------------

    bm25_results = search_bm25(
        collection=collection,
        query=query,
        source_name=source_name,
        top_k=max(top_k * 4, 20)
    )


    # ---------------------------------------------------------
    # 3. Normalize semantic scores
    # ---------------------------------------------------------

    semantic_candidates = {}

    semantic_documents = semantic_results[
        "documents"
    ][0]

    semantic_metadatas = semantic_results[
        "metadatas"
    ][0]

    semantic_ids = semantic_results[
        "ids"
    ][0]

    semantic_distances = semantic_results[
        "distances"
    ][0]


    semantic_scores = []

    for distance in semantic_distances:

        similarity = 1 - distance

        semantic_scores.append(
            similarity
        )


    if semantic_scores:

        min_score = min(
            semantic_scores
        )

        max_score = max(
            semantic_scores
        )

    else:

        min_score = 0
        max_score = 1


    for i in range(
        len(semantic_ids)
    ):

        score = semantic_scores[i]


        if max_score > min_score:

            normalized_score = (
                score - min_score
            ) / (
                max_score - min_score
            )

        else:

            normalized_score = 1.0


        semantic_candidates[
            semantic_ids[i]
        ] = {
            "id": semantic_ids[i],
            "document": semantic_documents[i],
            "metadata": semantic_metadatas[i],
            "semantic_score": normalized_score,
            "bm25_score": 0.0
        }


    # ---------------------------------------------------------
    # 4. Normalize BM25 scores
    # ---------------------------------------------------------

    bm25_scores = [
        result["bm25_score"]
        for result in bm25_results
    ]


    if bm25_scores:

        min_bm25 = min(
            bm25_scores
        )

        max_bm25 = max(
            bm25_scores
        )

    else:

        min_bm25 = 0
        max_bm25 = 1


    for result in bm25_results:

        result_id = result[
            "id"
        ]

        score = result[
            "bm25_score"
        ]


        if max_bm25 > min_bm25:

            normalized_score = (
                score - min_bm25
            ) / (
                max_bm25 - min_bm25
            )

        else:

            normalized_score = 1.0


        if result_id in semantic_candidates:

            semantic_candidates[
                result_id
            ][
                "bm25_score"
            ] = normalized_score

        else:

            semantic_candidates[
                result_id
            ] = {
                "id": result_id,
                "document": result[
                    "document"
                ],
                "metadata": result[
                    "metadata"
                ],
                "semantic_score": 0.0,
                "bm25_score": normalized_score
            }


    # ---------------------------------------------------------
    # 5. Calculate hybrid score
    # ---------------------------------------------------------

    candidates = []

    for candidate in semantic_candidates.values():

        hybrid_score = (
            semantic_weight
            * candidate[
                "semantic_score"
            ]
            +
            bm25_weight
            * candidate[
                "bm25_score"
            ]
        )

        candidate[
            "hybrid_score"
        ] = hybrid_score

        candidates.append(
            candidate
        )


    # ---------------------------------------------------------
    # 6. Sort by hybrid score
    # ---------------------------------------------------------

    candidates.sort(
        key=lambda x: x[
            "hybrid_score"
        ],
        reverse=True
    )


    # ---------------------------------------------------------
    # 7. Select diverse pages
    # ---------------------------------------------------------

    selected = []

    seen_pages = set()


    for candidate in candidates:

        metadata = candidate[
            "metadata"
        ]

        source = metadata.get(
            "source",
            source_name
        )

        page = metadata.get(
            "page_number",
            None
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


    # ---------------------------------------------------------
    # 8. Return Chroma-compatible structure
    # ---------------------------------------------------------

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
            1 - candidate[
                "hybrid_score"
            ]
            for candidate in selected
        ]],

        "semantic_scores": [[
            candidate[
                "semantic_score"
            ]
            for candidate in selected
        ]],

        "bm25_scores": [[
            candidate[
                "bm25_score"
            ]
            for candidate in selected
        ]],

        "hybrid_scores": [[
            candidate[
                "hybrid_score"
            ]
            for candidate in selected
        ]]
    }


def rerank_results(
    query,
    results,
    top_k=5
):
    """
    Rerank retrieved candidates using a
    cross-encoder model.

    The cross-encoder evaluates the query
    and each retrieved document together.

    After reranking, duplicate pages are removed
    to improve evidence diversity.
    """

    if not results:
        return []


    # ---------------------------------------------------------
    # 1. Create query-document pairs
    # ---------------------------------------------------------

    pairs = []

    for result in results:

        pairs.append(
            (
                query,
                result["document"]
            )
        )


    # ---------------------------------------------------------
    # 2. Generate reranker scores
    # ---------------------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # ---------------------------------------------------------
    # 3. Attach reranker scores
    # ---------------------------------------------------------

    reranked = []

    for i, result in enumerate(
        results
    ):

        result_copy = result.copy()

        result_copy[
            "reranker_score"
        ] = float(
            scores[i]
        )

        reranked.append(
            result_copy
        )


    # ---------------------------------------------------------
    # 4. Sort by reranker score
    # ---------------------------------------------------------

    reranked.sort(
        key=lambda x: x[
            "reranker_score"
        ],
        reverse=True
    )


    # ---------------------------------------------------------
    # 5. Select diverse pages
    # ---------------------------------------------------------

    selected = []

    seen_pages = set()


    for result in reranked:

        metadata = result[
            "metadata"
        ]

        source = metadata.get(
            "source",
            "unknown"
        )

        page = metadata.get(
            "page_number",
            None
        )

        page_key = (
            source,
            page
        )


        # Avoid retrieving multiple chunks
        # from the same page.
        if page_key in seen_pages:
            continue


        seen_pages.add(
            page_key
        )

        selected.append(
            result
        )


        if len(selected) >= top_k:
            break


    # ---------------------------------------------------------
    # 6. Return final reranked results
    # ---------------------------------------------------------

    return selected
    """
    Rerank retrieved candidates using a
    cross-encoder model.

    The cross-encoder evaluates the query
    and each retrieved document together.
    """

    if not results:

        return []


    # ---------------------------------------------------------
    # Create query-document pairs
    # ---------------------------------------------------------

    pairs = []

    for result in results:

        pairs.append(
            (
                query,
                result["document"]
            )
        )


    # ---------------------------------------------------------
    # Generate reranker scores
    # ---------------------------------------------------------

    scores = reranker.predict(
        pairs
    )


    # ---------------------------------------------------------
    # Attach scores
    # ---------------------------------------------------------

    reranked = []

    for i, result in enumerate(
        results
    ):

        result_copy = result.copy()

        result_copy[
            "reranker_score"
        ] = float(
            scores[i]
        )

        reranked.append(
            result_copy
        )


    # ---------------------------------------------------------
    # Sort by reranker score
    # ---------------------------------------------------------

    reranked.sort(
        key=lambda x: x[
            "reranker_score"
        ],
        reverse=True
    )


    # ---------------------------------------------------------
    # Return top results
    # ---------------------------------------------------------

    return reranked[
        :top_k
    ]


def search_hybrid_reranked(
    collection,
    query,
    source_name=None,
    top_k=5,
    candidate_k=20,
    semantic_weight=0.6,
    bm25_weight=0.4,
    rerank_query=None
):
    """
    Perform the complete retrieval pipeline:

    1. Semantic retrieval
    2. BM25 retrieval
    3. Hybrid score calculation
    4. CrossEncoder reranking
    5. Page-diverse final selection

    This function is the main high-quality retrieval
    entry point for the RAG system.
    """

    # ---------------------------------------------------------
    # 1. Hybrid retrieval
    # ---------------------------------------------------------

    hybrid_results = search_hybrid(
        collection=collection,
        query=query,
        source_name=source_name,
        top_k=candidate_k,
        semantic_weight=semantic_weight,
        bm25_weight=bm25_weight
    )


    documents = hybrid_results[
        "documents"
    ][0]

    metadatas = hybrid_results[
        "metadatas"
    ][0]

    ids = hybrid_results[
        "ids"
    ][0]

    semantic_scores = hybrid_results[
        "semantic_scores"
    ][0]

    bm25_scores = hybrid_results[
        "bm25_scores"
    ][0]

    hybrid_scores = hybrid_results[
        "hybrid_scores"
    ][0]


    if not documents:

        return {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "semantic_scores": [[]],
            "bm25_scores": [[]],
            "hybrid_scores": [[]],
            "reranker_scores": [[]]
        }


    # ---------------------------------------------------------
    # 2. Build reranker candidates
    # ---------------------------------------------------------

    candidates = []
    for i in range(len(documents)):
        candidates.append({
            "id": ids[i],
            "document": documents[i],
            "metadata": metadatas[i],
            "semantic_score": semantic_scores[i],
            "bm25_score": bm25_scores[i],
            "hybrid_score": hybrid_scores[i]
    })

    # ---------------------------------------------------------
    # 3. CrossEncoder reranking
    # ---------------------------------------------------------

    reranked = rerank_results(
        query=rerank_query or query,
        results=candidates,
        top_k=top_k
    )


    # ---------------------------------------------------------
    # 4. Convert to Chroma-compatible structure
    # ---------------------------------------------------------

    return {
        "ids": [[
            result["id"]
            for result in reranked
        ]],

        "documents": [[
            result["document"]
            for result in reranked
        ]],

        "metadatas": [[
            result["metadata"]
            for result in reranked
        ]],

        "distances": [[
            1 - result[
                "reranker_score"
            ]
            for result in reranked
        ]],

        "semantic_scores": [[
            result.get(
                "semantic_score",
                0.0
            )
            for result in reranked
        ]],

        "bm25_scores": [[
            result.get(
                "bm25_score",
                0.0
            )
            for result in reranked
        ]],

        "hybrid_scores": [[
            result.get(
                "hybrid_score",
                0.0
            )
            for result in reranked
        ]],

        "reranker_scores": [[
            result.get(
                "reranker_score",
                0.0
            )
            for result in reranked
        ]]
    }
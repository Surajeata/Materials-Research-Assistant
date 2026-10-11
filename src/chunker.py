import re


# ============================================================
# TEXT CLEANING
# ============================================================


def clean_page_text(text):
    """Normalize PDF-extracted text before chunking."""

    if not text:
        return ""

    # Normalize Windows and Mac line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Repair words split by a hyphen at a line boundary.
    # Example: defor-\nmation -> deformation
    text = re.sub(
        r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[A-Za-z])",
        "",
        text,
    )

    # Join ordinary PDF line wrapping with spaces.
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)

    # Collapse repeated spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()



# ============================================================
# SENTENCE SPLITTING
# ============================================================

def split_into_sentences(text):
    """
    Split cleaned text into approximately sentence-sized
    pieces.

    This is intentionally lightweight.
    """

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    cleaned_sentences = []

    for sentence in sentences:

        sentence = sentence.strip()

        if sentence:
            cleaned_sentences.append(
                sentence
            )

    return cleaned_sentences


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(
    pages,
    source_name="document",
    document_id="unknown",
    chunk_size=1200,
    overlap=200
):
    """
    Create intelligent overlapping chunks from PDF pages.

    Each chunk contains:

        chunk_id
        source
        document_id
        page_number
        text
        is_reference

    Parameters
    ----------
    pages : list
        Output from extract_text_from_pdf()

    source_name : str
        PDF filename.

    document_id : str
        Unique document identifier.

    chunk_size : int
        Approximate maximum number of characters
        per chunk.

    overlap : int
        Approximate overlap between adjacent chunks.
    """

    chunks = []

    # ---------------------------------------------------------
    # Validate parameters
    # ---------------------------------------------------------

    if chunk_size <= 0:

        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0:

        raise ValueError(
            "overlap cannot be negative."
        )

    if overlap >= chunk_size:

        raise ValueError(
            "overlap must be smaller than chunk_size."
        )

    # ---------------------------------------------------------
    # Process every page separately
    # ---------------------------------------------------------

    for page in pages:

        raw_text = page.get(
            "text",
            ""
        )

        page_number = page.get(
            "page_number",
            "Unknown"
        )

        is_reference = page.get(
            "is_reference",
            False
        )

        # -----------------------------------------------------
        # Clean extracted PDF text
        # -----------------------------------------------------

        text = clean_page_text(
            raw_text
        )

        if not text:
            continue

        # -----------------------------------------------------
        # Split into sentences
        # -----------------------------------------------------

        sentences = split_into_sentences(
            text
        )

        if not sentences:
            continue

        # -----------------------------------------------------
        # Build chunks
        # -----------------------------------------------------

        current_sentences = []

        current_length = 0

        chunk_number = 1

        for sentence in sentences:

            sentence_length = len(
                sentence
            )

            # -------------------------------------------------
            # If adding the sentence would exceed the chunk
            # size, save the current chunk first.
            # -------------------------------------------------

            if (
                current_sentences
                and
                current_length
                + sentence_length
                + 1
                > chunk_size
            ):

                chunk_text = " ".join(
                    current_sentences
                ).strip()

                if chunk_text:

                    chunks.append({
                        "chunk_id": (
                            f"{document_id}_"
                            f"page_{page_number}_"
                            f"chunk_{chunk_number}"
                        ),

                        "source": source_name,

                        "document_id": document_id,

                        "page_number": page_number,

                        "text": chunk_text,

                        "is_reference": is_reference
                    })

                    chunk_number += 1

                # -------------------------------------------------
                # Create overlapping sentences.
                #
                # Keep sentences from the end of the previous
                # chunk until approximately `overlap` characters
                # are reached.
                # -------------------------------------------------

                overlap_sentences = []

                overlap_length = 0

                for previous_sentence in reversed(
                    current_sentences
                ):

                    sentence_size = (
                        len(previous_sentence)
                        + 1
                    )

                    if (
                        overlap_length
                        + sentence_size
                        > overlap
                    ):
                        break

                    overlap_sentences.insert(
                        0,
                        previous_sentence
                    )

                    overlap_length += (
                        sentence_size
                    )

                current_sentences = (
                    overlap_sentences
                )

                current_length = (
                    overlap_length
                )

            # -------------------------------------------------
            # Add current sentence
            # -------------------------------------------------

            current_sentences.append(
                sentence
            )

            current_length += (
                sentence_length + 1
            )

        # -----------------------------------------------------
        # Save final chunk from this page
        # -----------------------------------------------------

        if current_sentences:

            chunk_text = " ".join(
                current_sentences
            ).strip()

            if chunk_text:

                chunks.append({
                    "chunk_id": (
                        f"{document_id}_"
                        f"page_{page_number}_"
                        f"chunk_{chunk_number}"
                    ),

                    "source": source_name,

                    "document_id": document_id,

                    "page_number": page_number,

                    "text": chunk_text,

                    "is_reference": is_reference
                })

    return chunks
import hashlib
from pathlib import Path

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import (
    create_collection,
    store_chunks,
    document_exists
)


def calculate_document_id(pdf_path):
    """
    Create a unique SHA-256 fingerprint for a PDF.
    """

    sha256 = hashlib.sha256()

    with open(pdf_path, "rb") as file:

        while True:

            data = file.read(1024 * 1024)

            if not data:
                break

            sha256.update(data)

    return sha256.hexdigest()


def ingest_pdf(pdf_path):
    """
    Process a PDF and store it in ChromaDB.

    Returns:
        dictionary containing processing status.
    """

    pdf_path = Path(pdf_path)

    source_name = pdf_path.name

    document_id = calculate_document_id(
        pdf_path
    )

    collection = create_collection()

    # Check whether this PDF is already indexed
    if document_exists(
        collection,
        document_id
    ):

        return {
            "status": "already_exists",
            "source": source_name,
            "document_id": document_id,
            "chunks": 0
        }

    # Extract PDF text
    pages = extract_text_from_pdf(
        str(pdf_path)
    )

    # Create chunks
    chunks = create_chunks(
        pages,
        source_name=source_name,
        document_id=document_id
    )

    # Create embeddings
    embeddings = create_embeddings(
        chunks
    )

    # Store in ChromaDB
    store_chunks(
        collection,
        chunks,
        embeddings
    )

    return {
        "status": "processed",
        "source": source_name,
        "document_id": document_id,
        "chunks": len(chunks)
    }
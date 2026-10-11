import chromadb


DB_PATH = "data/chroma_db"


def create_collection():
    """
    Create or load the ChromaDB collection.
    """

    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name="research_papers"
    )

    return collection


def document_exists(collection, document_id):
    """
    Check whether a document has already been indexed.
    """

    results = collection.get(
        where={
            "document_id": document_id
        },
        limit=1
    )

    return len(results["ids"]) > 0


def store_chunks(collection, chunks, embeddings):
    """
    Store chunks, embeddings, and metadata in ChromaDB.
    """

    ids = [
        chunk["chunk_id"]
        for chunk in chunks
    ]

    documents = [
        chunk["text"]
        for chunk in chunks
    ]

    metadatas = [
    {
        "source": chunk["source"],
        "page_number": chunk["page_number"],
        "document_id": chunk["document_id"],
        "is_reference": chunk["is_reference"]
    }
    for chunk in chunks
]

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(
        f"Stored {len(chunks)} chunks in ChromaDB."
    )


def delete_document(collection, document_id):
    """
    Delete all chunks belonging to a document.
    """

    collection.delete(
        where={
            "document_id": document_id
        }
    )
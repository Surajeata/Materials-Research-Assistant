from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_collection, store_chunks


PDF_PATH = "data/documents/research_paper.pdf"


# Step 1: Extract PDF
pages = extract_text_from_pdf(PDF_PATH)


# Step 2: Create chunks
chunks = create_chunks(pages)


# Step 3: Create embeddings
embeddings = create_embeddings(chunks)


# Step 4: Create ChromaDB collection
collection = create_collection()


# Step 5: Store everything
store_chunks(collection, chunks, embeddings)


print("\n" + "=" * 70)
print("VECTOR DATABASE TEST")
print("=" * 70)

print("Collection:", collection.name)
print("Number of stored documents:", collection.count())
from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings
from src.vector_store import create_collection
from src.retriever import search_collection


PDF_PATH = "data/documents/research_paper.pdf"


# Load the PDF
pages = extract_text_from_pdf(PDF_PATH)


# Create chunks
chunks = create_chunks(pages)


# Create embeddings
embeddings = create_embeddings(chunks)


# Connect to existing ChromaDB
collection = create_collection()


query = "How do rare-earth elements affect texture evolution in magnesium alloys?"


results = search_collection(
    collection,
    query,
    top_k=5
)


print("\n" + "=" * 80)
print("SEMANTIC SEARCH RESULTS")
print("=" * 80)

print(f"\nQuery: {query}")


for i in range(len(results["documents"][0])):

    print("\n" + "-" * 80)

    print(f"Result #{i + 1}")

    print(f"Page: {results['metadatas'][0][i]['page_number']}")

    print(f"Chunk ID: {results['ids'][0][i]}")

    print("\nText:")

    print(results["documents"][0][i][:1000])
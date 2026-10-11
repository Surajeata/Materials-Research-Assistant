from src.vector_store import create_collection
from src.rag import answer_question


# Connect to our existing ChromaDB
collection = create_collection()


question = """
How do rare-earth elements affect texture evolution
during processing of magnesium alloys?
"""


answer, results = answer_question(
    collection,
    question,
    top_k=5
)


print("\n" + "=" * 80)
print("RAG ANSWER")
print("=" * 80)

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(answer)


print("\n" + "=" * 80)
print("RETRIEVED SOURCES")
print("=" * 80)


for i, chunk_id in enumerate(results["ids"][0]):
    metadata = results["metadatas"][0][i]
    document = results["documents"][0][i]

    print(f"\nSOURCE {i + 1}")
    print(f"ID: {chunk_id}")
    print(f"Source: {metadata.get('source')}")
    print(f"Page: {metadata.get('page_number')}")
    print("\nRetrieved text:")
    print(document[:1500])
    print("-" * 80)
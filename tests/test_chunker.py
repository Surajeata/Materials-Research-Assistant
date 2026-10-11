from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks


PDF_PATH = "data/documents/A review on the effect of rare-earth elements on texture evolution during processing of magnesium alloys.pdf"

pages = extract_text_from_pdf(PDF_PATH)

chunks = create_chunks(pages)


print(f"Pages: {len(pages)}")
print(f"Chunks: {len(chunks)}")


for chunk in chunks[:10]:
    print("\n" + "=" * 70)
    print(f"Chunk ID: {chunk['chunk_id']}")
    print(f"Page: {chunk['page_number']}")
    print(f"Length: {len(chunk['text'])}")
    print("=" * 70)
    print(chunk["text"][:500])
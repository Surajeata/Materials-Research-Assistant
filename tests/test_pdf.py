from src.pdf_processor import extract_text_from_pdf


PDF_PATH = "data/documents/A review on the effect of rare-earth elements on texture evolution during processing of magnesium alloys.pdf"

pages = extract_text_from_pdf(PDF_PATH)

print(f"Total pages extracted: {len(pages)}")

for page in pages[:2]:

    print("\n" + "=" * 60)
    print(f"PAGE {page['page_number']}")
    print("=" * 60)

    print(page["text"][:1000])
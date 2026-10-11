import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parents[1]
    )
)

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks


DOCUMENTS_DIR = Path(
    "data/documents"
)


pdf_files = list(
    DOCUMENTS_DIR.glob("*.pdf")
)


print()
print("=" * 80)
print("INTELLIGENT CHUNKING TEST")
print("=" * 80)


for pdf_path in pdf_files:

    print()
    print(
        "=" * 80
    )

    print(
        f"FILE: {pdf_path.name}"
    )

    print(
        "=" * 80
    )


    pages = extract_text_from_pdf(
        str(pdf_path)
    )


    chunks = create_chunks(
        pages,
        source_name=pdf_path.name,
        document_id="test"
    )


    print(
        f"\nPages: {len(pages)}"
    )

    print(
        f"Chunks: {len(chunks)}"
    )


    # ---------------------------------------------------------
    # Show first five chunks
    # ---------------------------------------------------------

    for i, chunk in enumerate(
        chunks[:5],
        start=1
    ):

        print()
        print(
            f"CHUNK {i}"
        )

        print(
            f"Page: "
            f"{chunk['page_number']}"
        )

        print(
            f"Length: "
            f"{len(chunk['text'])}"
        )

        print(
            "Text:"
        )

        print(
            chunk["text"][:1000]
        )

        print(
            "-" * 80
        )
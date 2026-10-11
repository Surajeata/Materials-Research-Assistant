from pathlib import Path
import sys
import numpy as np

# Allow imports when running this file directly
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1])
)

from src.pdf_processor import extract_text_from_pdf
from src.chunker import create_chunks
from src.embeddings import create_embeddings


DOCUMENTS_DIR = Path("data/documents")


def main():
    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"ERROR: No PDF files found in {DOCUMENTS_DIR.resolve()}")
        return

    print("=" * 75)
    print("EMBEDDING QUALITY DIAGNOSTIC")
    print("=" * 75)

    overall_pass = True

    for pdf_path in pdf_files:
        print(f"\nPDF: {pdf_path.name}")
        print("-" * 75)

        try:
            pages = extract_text_from_pdf(str(pdf_path))
            chunks = create_chunks(
                pages,
                source_name=pdf_path.name,
                document_id="embedding_diagnostic"
            )

            if not chunks:
                print("FAIL: No chunks generated.")
                overall_pass = False
                continue

            embeddings = create_embeddings(chunks)

            if embeddings is None or len(embeddings) == 0:
                print("FAIL: No embeddings generated.")
                overall_pass = False
                continue

            vectors = np.asarray(embeddings, dtype=np.float64)

            if vectors.ndim != 2:
                print(f"FAIL: Unexpected embedding shape: {vectors.shape}")
                overall_pass = False
                continue

            norms = np.linalg.norm(vectors, axis=1)
            finite_rows = np.all(np.isfinite(vectors), axis=1)
            nonzero_rows = norms > 1e-12

            checks = {
                "Chunks generated": len(chunks) > 0,
                "Chunk/embedding counts match": len(chunks) == len(embeddings),
                "Embedding vectors have consistent dimensions": vectors.shape[1] > 0,
                "All embedding values are finite": bool(np.all(finite_rows)),
                "All embeddings are non-zero": bool(np.all(nonzero_rows)),
            }

            print(f"Pages extracted: {len(pages)}")
            print(f"Chunks generated: {len(chunks)}")
            print(f"Embeddings generated: {len(embeddings)}")
            print(f"Embedding dimensions: {vectors.shape[1]}")
            print(f"Average vector norm: {norms.mean():.6f}")
            print(f"Minimum vector norm: {norms.min():.6f}")
            print(f"Maximum vector norm: {norms.max():.6f}")

            print("\nValidation:")
            for name, passed in checks.items():
                print(f"{'PASS' if passed else 'FAIL'}: {name}")

            if not all(checks.values()):
                overall_pass = False

            print("\nFirst chunk preview:")
            print(chunks[0]["text"][:350])

        except Exception as exc:
            overall_pass = False
            print(f"ERROR: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 75)
    print("OVERALL RESULT:", "PASS" if overall_pass else "CHECK FAILURES")
    print("=" * 75)


if __name__ == "__main__":
    main()

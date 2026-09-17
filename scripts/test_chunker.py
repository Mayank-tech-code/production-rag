from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import normalize_whitespace
from app.chunking.recursive_chunker import split_documents


PDF_PATH = "data/raw/resume.pdf"


documents = load_pdf(PDF_PATH)

# Clean documents
for document in documents:
    document.page_content = normalize_whitespace(
        document.page_content
    )

# Create chunks
chunks = split_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=150,
)

print(f"Total documents/pages: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

for index, chunk in enumerate(chunks[:5]):

    print("\n" + "=" * 60)
    print(f"CHUNK {index + 1}")
    print("=" * 60)

    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)
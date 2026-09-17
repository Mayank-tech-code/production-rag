from app.ingestion.loader import load_pdf


PDF_PATH = "data/raw/resume.pdf"


documents = load_pdf(PDF_PATH)

print(f"Total pages loaded: {len(documents)}")

for document in documents[:2]:
    print("\n--- PAGE ---")
    print(document.page_content[:1000])
    print("\nMetadata:")
    print(document.metadata)
from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import normalize_whitespace


PDF_PATH = "data/raw/resume.pdf"


documents = load_pdf(PDF_PATH)

for document in documents[:2]:

    original_text = document.page_content

    cleaned_text = normalize_whitespace(original_text)

    print("\n==============================")
    print("ORIGINAL TEXT")
    print("==============================")
    print(original_text[:1500])

    print("\n==============================")
    print("CLEANED TEXT")
    print("==============================")
    print(cleaned_text[:1500])
from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import normalize_whitespace
from app.chunking.recursive_chunker import split_documents

from app.config.settings import (
    QDRANT_URL,
    QDRANT_COLLECTION,
)

from app.embeddings.embedding_model import get_embedding_model

from langchain_qdrant import QdrantVectorStore


PDF_PATH = "data/raw/resume.pdf"


def main():

    # 1. Load
    documents = load_pdf(PDF_PATH)

    # 2. Clean
    for document in documents:
        document.page_content = normalize_whitespace(
            document.page_content
        )

    # 3. Chunk
    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=150,
    )

    print(f"Documents/pages: {len(documents)}")
    print(f"Chunks: {len(chunks)}")

    # 4. Embedding model
    embedding_model = get_embedding_model()

    # 5. Create collection and index documents
    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
    )

    print("Documents successfully indexed into Qdrant.")


if __name__ == "__main__":
    main()
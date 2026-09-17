from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import normalize_whitespace
from app.chunking.recursive_chunker import split_documents
from app.embeddings.embedding_model import get_embedding_model


PDF_PATH = "data/raw/resume.pdf"


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

# 4. Embedding model
embedding_model = get_embedding_model()

# 5. Embed first chunk
first_chunk = chunks[0]

vector = embedding_model.embed_query(
    first_chunk.page_content
)

print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(first_chunk.page_content)

print("\nMetadata:")
print(first_chunk.metadata)

print("\nVector dimensions:")
print(len(vector))

print("\nFirst 10 vector values:")
print(vector[:10])
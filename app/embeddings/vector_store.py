from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config.settings import (
    QDRANT_URL,
    QDRANT_COLLECTION,
)
from app.embeddings.embedding_model import get_embedding_model


def get_qdrant_client():
    return QdrantClient(
        url=QDRANT_URL
    )


def get_vector_store():
    client = get_qdrant_client()

    embedding_model = get_embedding_model()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION,
        embedding=embedding_model,
    )

    return vector_store
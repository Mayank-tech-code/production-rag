from langchain_qdrant import QdrantVectorStore

from app.config.settings import (
    QDRANT_URL,
    QDRANT_COLLECTION,
)

from app.embeddings.embedding_model import get_embedding_model


def get_vector_store():

    embedding_model = get_embedding_model()

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embedding_model,
        collection_name=QDRANT_COLLECTION,
        url=QDRANT_URL,
    )

    return vector_store


def retrieve_documents_with_scores(
    query: str,
    k: int = 5,
):

    vector_store = get_vector_store()

    return vector_store.similarity_search_with_score(
        query,
        k=k,
    )


def retrieve_relevant_documents(
    query: str,
    k: int = 5,
    score_threshold: float = 0.4,
):

    results = retrieve_documents_with_scores(
        query=query,
        k=k,
    )

    relevant_documents = [
        document
        for document, score in results
        if score >= score_threshold
    ]

    return relevant_documents
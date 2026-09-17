from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from app.ingestion.loader import load_pdf
from app.ingestion.cleaner import normalize_whitespace
from app.chunking.recursive_chunker import split_documents

from app.retrieval.retriever import (
    retrieve_documents_with_scores,
)


PDF_PATH = "data/raw/resume.pdf"


def load_chunks() -> list[Document]:

    documents = load_pdf(PDF_PATH)

    for document in documents:
        document.page_content = normalize_whitespace(
            document.page_content
        )

    chunks = split_documents(
        documents,
        chunk_size=1000,
        chunk_overlap=150,
    )

    return chunks


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    rrf_k: int = 60,
) -> list[Document]:

    scores = {}
    documents = {}

    for results in result_lists:

        for rank, document in enumerate(
            results,
            start=1,
        ):

            key = (
                document.metadata.get("source", ""),
                document.metadata.get("page", ""),
                document.page_content,
            )

            documents[key] = document

            scores[key] = scores.get(key, 0) + (
                1 / (rrf_k + rank)
            )

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[key]
        for key in ranked_keys
    ]


def hybrid_search(
    query: str,
    k: int = 5,
) -> list[Document]:

    # --------------------------------
    # 1. Dense / Semantic Search
    # --------------------------------

    dense_results = retrieve_documents_with_scores(
        query=query,
        k=10,
    )

    dense_documents = [
        document
        for document, score in dense_results
    ]

    # --------------------------------
    # 2. BM25 Keyword Search
    # --------------------------------

    chunks = load_chunks()

    bm25_retriever = BM25Retriever.from_documents(
        chunks
    )

    bm25_retriever.k = 10

    keyword_documents = bm25_retriever.invoke(
        query
    )

    # --------------------------------
    # 3. RRF Fusion
    # --------------------------------

    fused_results = reciprocal_rank_fusion(
        [
            dense_documents,
            keyword_documents,
        ]
    )

    return fused_results[:k]
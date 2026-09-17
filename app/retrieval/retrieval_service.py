from langchain_core.documents import Document

from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import DocumentReranker
from app.retrieval.context_optimizer import (
    deduplicate_documents,
    select_context,
)


class RetrievalService:

    def __init__(
        self,
        retrieval_k: int = 10,
        rerank_k: int = 10,
        context_k: int = 5,
    ):
        self.retrieval_k = retrieval_k
        self.rerank_k = rerank_k
        self.context_k = context_k

        self.reranker = DocumentReranker()

    def retrieve(self, query: str) -> list[Document]:

        # 1. Hybrid retrieval
        candidates = hybrid_search(
            query=query,
            k=self.retrieval_k,
        )

        # 2. Reranking
        reranked_results = self.reranker.rerank(
            query=query,
            documents=candidates,
            top_k=self.rerank_k,
        )

        # Remove scores because the next stage only needs documents
        reranked_documents = [
            document
            for document, score in reranked_results
        ]

        # 3. Deduplication
        unique_documents = deduplicate_documents(
            reranked_documents
        )

        # 4. Context selection
        final_context = select_context(
            unique_documents,
            max_documents=self.context_k,
        )

        return final_context
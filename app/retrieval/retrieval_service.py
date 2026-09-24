from httpx2 import query
from langchain_core.documents import Document

from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import DocumentReranker
from app.retrieval.context_optimizer import (
    deduplicate_documents,
    select_context,
)
from app.retrieval.query_rewriter import rewrite_query 


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

        # 1. Rewrite / normalize query
        rewritten_query = rewrite_query(query)

        # 2. Hybrid retrieval
        candidates = hybrid_search(
            query=rewritten_query,
            k=self.retrieval_k
        )

        # 3. Rerank
        reranked_results = self.reranker.rerank(
            query=rewritten_query,
            documents=candidates,
            top_k=self.rerank_k
        )

        # 4. Extract documents
        reranked_documents = [
            document
            for document, score in reranked_results
        ]

        # 5. Remove duplicates
        unique_documents = deduplicate_documents(
            reranked_documents
        )

        # 6. Select final context
        return select_context(
            unique_documents,
            max_documents=self.context_k
        )
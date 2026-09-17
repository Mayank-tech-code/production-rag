from sentence_transformers import CrossEncoder
from langchain_core.documents import Document


MODEL_NAME = "BAAI/bge-reranker-base"


class DocumentReranker:

    def __init__(self):

        self.model = CrossEncoder(
            MODEL_NAME
        )

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int = 5,
    ) -> list[tuple[Document, float]]:

        if not documents:
            return []

        pairs = [
            (
                query,
                document.page_content,
            )
            for document in documents
        ]

        scores = self.model.predict(
            pairs
        )

        ranked_results = sorted(
            zip(documents, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        return ranked_results[:top_k]
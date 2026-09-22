from app.retrieval.retrieval_service import RetrievalService
from app.generation.prompt import build_prompt
from app.generation.llm import get_llm_client


class RAGService:

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.llm_client = get_llm_client()

    def answer(self, query: str):

        # 1. Retrieve relevant documents
        documents = self.retrieval_service.retrieve(query)

        # 2. Build prompt
        prompt = build_prompt(
            query=query,
            documents=documents
        )

        # 3. Generate answer
        response = self.llm_client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=512,
            temperature=0.1,
        )

        answer = response.choices[0].message.content

        # 4. Build citation metadata
        sources = []

        for document in documents:

            source = document.metadata.get(
                "source",
                "Unknown"
            )

            page = document.metadata.get(
                "page",
                None
            )

            if page is not None:
                page = int(page) + 1

            citation = {
                "source": source,
                "page": page
            }

            if citation not in sources:
                sources.append(citation)

        return {
            "answer": answer,
            "sources": sources
        }
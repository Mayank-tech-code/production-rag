from app.retrieval.retrieval_service import RetrievalService
from app.generation.prompt import build_prompt
from app.generation.llm import get_llm_client


class RAGService:

    def __init__(self):
        self.retrieval_service = RetrievalService()
        self.llm_client = get_llm_client()

    def answer(self, query: str) -> str:

        # 1. Retrieve relevant context
        documents = self.retrieval_service.retrieve(query)

        # 2. Build prompt
        prompt = build_prompt(
            query=query,
            documents=documents,
        )

        # 3. Generate answer
        response = self.llm_client.chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            max_tokens=512,
            temperature=0.1,
        )

        return response.choices[0].message.content
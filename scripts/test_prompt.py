from app.retrieval.retrieval_service import RetrievalService
from app.generation.prompt import build_prompt


query = "What frontend technologies does the candidate have experience with?"


retrieval_service = RetrievalService(
    retrieval_k=3,
    rerank_k=3,
    context_k=3,
)

documents = retrieval_service.retrieve(query)


prompt = build_prompt(
    query=query,
    documents=documents,
)


print(prompt)
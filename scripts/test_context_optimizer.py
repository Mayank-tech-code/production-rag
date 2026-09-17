from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import DocumentReranker
from app.retrieval.context_optimizer import (
    deduplicate_documents,
    select_context,
)


query = "What frontend technologies does the candidate have experience with?"


# 1. Hybrid retrieval
candidates = hybrid_search(
    query=query,
    k=10,
)

print(f"Hybrid candidates: {len(candidates)}")


# 2. Reranking
reranker = DocumentReranker()

reranked_results = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=4,
)

reranked_documents = [
    document
    for document, score in reranked_results
]

print(f"Reranked documents: {len(reranked_documents)}")


# 3. Deduplication
unique_documents = deduplicate_documents(
    reranked_documents
)

print(f"After deduplication: {len(unique_documents)}")


# 4. Final context
final_context = select_context(
    unique_documents,
    max_documents=5,
)

print(f"Final context chunks: {len(final_context)}")


for index, document in enumerate(final_context):

    print("\n" + "=" * 70)
    print(f"FINAL CONTEXT {index + 1}")
    print("=" * 70)

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)
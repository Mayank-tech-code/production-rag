from app.retrieval.hybrid_retriever import hybrid_search
from app.retrieval.reranker import DocumentReranker


query = "What frontend technologies does the candidate have experience with?"


# 1. Retrieve candidates
candidates = hybrid_search(
    query=query,
    k=10,
)

print("=" * 80)
print("HYBRID SEARCH RESULTS")
print("=" * 80)

for index, document in enumerate(candidates):

    print("\n" + "-" * 70)
    print(f"CANDIDATE {index + 1}")
    print(document.page_content[:500])


# 2. Rerank candidates
reranker = DocumentReranker()

reranked_results = reranker.rerank(
    query=query,
    documents=candidates,
    top_k=5,
)


print("\n\n")
print("=" * 80)
print("RERANKED RESULTS")
print("=" * 80)


for index, (document, score) in enumerate(
    reranked_results
):

    print("\n" + "-" * 70)
    print(f"RANK {index + 1}")
    print(f"RERANK SCORE: {score}")

    print("\nContent:")
    print(document.page_content[:500])

    print("\nMetadata:")
    print(document.metadata)
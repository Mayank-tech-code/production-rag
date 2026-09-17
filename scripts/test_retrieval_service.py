from app.retrieval.retrieval_service import RetrievalService


query = "What frontend technologies does the candidate have experience with?"

retrieval_service = RetrievalService(
    retrieval_k=3,
    rerank_k=3,
    context_k=3,
)

final_context = retrieval_service.retrieve(query)


print(f"Final context chunks: {len(final_context)}")


for index, document in enumerate(final_context):

    print("\n" + "=" * 70)
    print(f"CONTEXT {index + 1}")
    print("=" * 70)

    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)
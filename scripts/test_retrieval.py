from app.retrieval.retriever import retrieve_documents_with_scores


query = "How many annual leave days are employees entitled to?"


results = retrieve_documents_with_scores(
    query=query,
    k=5,
)


print(f"Query: {query}")
print(f"\nRetrieved documents: {len(results)}")


for index, (document, score) in enumerate(results):

    print("\n" + "=" * 70)
    print(f"RESULT {index + 1}")
    print("=" * 70)

    print(f"\nSimilarity Score: {score}")

    print("\nContent:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)
from app.retrieval.hybrid_retriever import hybrid_search


queries = [
    "What frontend technologies does the candidate have experience with?",
    "Does the candidate have experience with RTK Query?",
    "How many annual leave days are employees entitled to?",
]


for query in queries:

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = hybrid_search(
        query=query,
        k=5,
    )

    for index, document in enumerate(results):

        print("\n" + "-" * 70)
        print(f"RESULT {index + 1}")

        print("\nContent:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)
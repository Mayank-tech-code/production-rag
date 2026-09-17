from app.generation.rag_service import RAGService


rag = RAGService()


query = "What frontend technologies does the candidate have experience with?"


answer = rag.answer(query)


print("\n" + "=" * 70)
print("QUESTION")
print("=" * 70)
print(query)

print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)
print(answer)
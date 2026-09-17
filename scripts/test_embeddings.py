from app.embeddings.embedding_model import get_embedding_model


embedding_model = get_embedding_model()

text = "Employees are entitled to 24 days of annual leave per year."

vector = embedding_model.embed_query(text)

print("Vector type:", type(vector))
print("Vector dimensions:", len(vector))

print("\nFirst 10 values:")
print(vector[:10])
from sentence_transformers import SentenceTransformer
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
def create_embeddings(texts):
    return model.encode(texts)
def create_query_embedding(question):
    return model.encode([question])[0]
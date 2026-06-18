import faiss
import numpy as np
import pickle
dimension = 384

index = faiss.IndexFlatL2(dimension)
documents = []
def add_documents(chunks, embeddings):

    global documents

    index.add(
        np.array(embeddings).astype("float32")
    )

    documents.extend(chunks)
def search(query_embedding, k=3):

    distances, indices = index.search(
        np.array([query_embedding]).astype("float32"),
        k
    )

    results = []

    for idx in indices[0]:

        if idx < len(documents):
            results.append(documents[idx])

    return results
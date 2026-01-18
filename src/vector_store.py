import numpy as np

class VectorStore:
    def __init__(self, embeddings, texts):
        self.embeddings = embeddings # it stores the embeddings as an instance variable
        self.texts = texts

    def search(self, query_embedding, top_k=5):
        scores = np.dot(self.embeddings, query_embedding)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.texts[i], float(scores[i])) for i in top_idx]

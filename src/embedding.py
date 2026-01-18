from sentence_transformers import SentenceTransformer


#model selection
model = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

def embed_texts(texts):
    return model.encode(texts, normalize_embeddings=True)

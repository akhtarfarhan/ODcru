def retrieve(query, embed_fn, vector_store, top_k=5):
    question_emb = embed_fn([query])[0]
    return vector_store.search(question_emb, top_k)
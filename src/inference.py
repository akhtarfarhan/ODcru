import pandas as pd
from data_loader import load_dataset
from chunking import chunk_text
from embedding import embed_texts
from vector_store import VectorStore

DATA_PATH = "C:\\Users\\akhta\\OneDrive\\Desktop\\ODcRU\\Class3\\cancer_semantic_search\\data\\raw\\CancerQA.csv"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.5   # below this = out of scope

def build_index():
    df = load_dataset(DATA_PATH)

    all_chunks = []
    for answer in df["Answer"].astype(str):
        chunks = chunk_text(answer)
        all_chunks.extend(chunks)

    embeddings = embed_texts(all_chunks)
    return VectorStore(embeddings, all_chunks)

def main():
    print("Building medical retrieval index...")
    vector_store = build_index()
    print("Index ready.\n")

    while True:
        query = input("Ask a cancer-related question (or type 'exit'): ").strip()

        if query.lower() == "exit":
            print("Exiting system.")
            break

        # Embed the user query
        query_embedding = embed_texts([query])[0]

        # Retrieve top-k results
        results = vector_store.search(query_embedding, top_k=TOP_K)

        max_score = max(score for _, score in results)

        if max_score < SIMILARITY_THRESHOLD:
            print("\nI don't know.")
            print("The question is outside the scope of the cancer medical dataset.\n")
            continue

        rows = []
        for idx, (text, score) in enumerate(results, start=1):
            if score >= SIMILARITY_THRESHOLD:
                rows.append({
                    "Rank": idx,
                    "Cosine Similarity": round(score, 4),
                    "Retrieved Text": text[:400] + "..."
                })

        df_results = pd.DataFrame(rows)

        print("\nRetrieved Medical Evidence:")
        print(df_results.to_string(index=False))
        print()

if __name__ == "__main__":
    main()

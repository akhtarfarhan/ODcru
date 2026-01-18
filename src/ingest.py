import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "C:\\Users\\akhta\\OneDrive\\Desktop\\ODcRU\\class4\\data\\raw\\CancerQA.csv"
VECTOR_DB_PATH = "C:\\Users\\akhta\\OneDrive\\Desktop\\ODcRU\\class4\\vectorstore"

def main():
    # 1. Load dataset
    df = pd.read_csv(DATA_PATH)
    texts = df["Answer"].astype(str).tolist()

    # 2. Chunk text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=60
    )
    documents = splitter.create_documents(texts)

    # 3. Load embedding model (medical)
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )

    # 4. Create FAISS vector store
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 5. Save to disk
    vectorstore.save_local(VECTOR_DB_PATH)

    print("FAISS vector database created successfully.")

if __name__ == "__main__":
    main()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

VECTOR_DB_PATH = "C:\\Users\\akhta\\OneDrive\\Desktop\\ODcRU\\class4\\vectorstore\\faiss_index"

def build_rag_chain():
    # 1. Load embeddings (must match ingest.py)
    embeddings = HuggingFaceEmbeddings(
        model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
    )

    # 2. Load FAISS vector store
    vectorstore = FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 3. HuggingFace LLM pipeline
    hf_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256
)

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # 4. Medical-safe prompt
    prompt = ChatPromptTemplate.from_template(
        """
        You are a medical assistant.
        Answer the question using ONLY the context below.
        If the answer is not present, say "I don't know."

        Context:
        {context}

        Question:
        {question}
        """
    )

    # 5. RAG pipeline using LCEL (Runnable)
    rag_pipeline = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_pipeline

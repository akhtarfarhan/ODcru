# CancerQA RAG System

A Retrieval-Augmented Generation (RAG) system built with LangChain and FAISS for answering cancer-related questions using the CancerQA dataset.

## Project Structure

```
.
├── config/
│   └── config.yaml              # Configuration settings
├── data/
│   ├── raw/
│   │   └── CancerQA.csv         # Original dataset
│   └── processed/
│       └── chunks.json          # Processed and chunked data
├── src/
│   ├── app.py                   # Main application
│   ├── data_loader.py           # Data loading utilities
│   ├── ingest.py                # Data ingestion pipeline
│   └── rag_chain.py             # RAG chain implementation
├── vectorstore/
│   └── faiss_index/
│       └── index.faiss          # FAISS vector index
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Installation

1. **Clone or download the project**

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config/config.yaml` to customize:
- Model settings
- Chunk size and overlap
- Vector store parameters
- Other system configurations

## Usage

### Data Ingestion

Process the raw CancerQA data and build the vector index:
```bash
python src/ingest.py
```

This will:
- Load the CSV data
- Chunk documents into manageable pieces
- Generate embeddings
- Store vectors in FAISS index

### Running the Application

Start the RAG system:
```bash
python src/app.py
```

The application will:
- Load the FAISS vector store
- Initialize the RAG chain
- Accept user queries about cancer-related topics
- Return answers retrieved from the knowledge base

## Key Components

- **data_loader.py**: Handles loading and parsing the CancerQA dataset
- **ingest.py**: Orchestrates document processing and vector store creation
- **rag_chain.py**: Implements the RAG pipeline combining retrieval and generation
- **app.py**: User-facing application interface

## Dependencies

See `requirements.txt` for the full list. Key dependencies include:
- LangChain
- FAISS
- OpenAI/LLM models
- PyYAML

## Notes

- The vector store and processed data are generated during ingestion
- FAISS index is stored locally in `vectorstore/faiss_index/`
- Configuration can be adjusted before running ingestion for optimal performance

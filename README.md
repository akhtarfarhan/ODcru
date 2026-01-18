# Cancer Semantic Search

A semantic search system for cancer-related medical questions using embeddings and vector similarity retrieval.

## Overview

This project implements a retrieval system that answers cancer-related questions by:
1. Loading medical Q&A data from a CSV dataset
2. Chunking text into manageable segments
3. Generating embeddings using sentence transformers
4. Storing embeddings in a vector database
5. Retrieving relevant answers via semantic similarity

## Project Structure

```
cancer_semantic_search/
├── data/
│   ├── raw/                 # Original data files
│   │   └── CancerQA.csv
│   └── processed/           # Processed/chunked data
│       └── chunks.json
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Load CSV data
│   ├── chunking.py          # Text chunking logic
│   ├── embedding.py         # Generate embeddings
│   ├── vector_store.py      # Vector storage and retrieval
│   ├── retriever.py         # Retrieval logic
│   └── inference.py         # Main inference pipeline
├── config/
│   └── config.yaml          # Configuration settings
├── vectorstore/             # Stored embeddings (generated)
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`:
  - numpy
  - pandas
  - sentence-transformers
  - torch
  - scikit-learn
  - tqdm

## Installation

1. Clone or download the project
2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main inference pipeline:

```bash
python src/inference.py
```

This will:
- Build an index from the cancer QA dataset
- Start an interactive session where you can ask cancer-related questions
- Return the most relevant answers with similarity scores

Type `exit` to quit the program.

## Key Components

- **data_loader.py**: Loads the CancerQA.csv dataset
- **chunking.py**: Splits long answers into smaller semantic chunks
- **embedding.py**: Generates embeddings using sentence transformers
- **vector_store.py**: Stores and retrieves embeddings using similarity search
- **retriever.py**: Core retrieval logic
- **inference.py**: Interactive query interface and main pipeline

## Configuration

Adjust settings in `src/inference.py`:
- `TOP_K`: Number of results to return (default: 3)
- `SIMILARITY_THRESHOLD`: Minimum similarity score (default: 0.5)

## Output

Generated artifacts (not tracked in git):
- `vectorstore/`: Vector embeddings and index files
- `data/processed/chunks.json`: Processed text chunks

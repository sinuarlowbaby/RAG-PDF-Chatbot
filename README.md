# RAG Application

This is a Retrieval-Augmented Generation (RAG) application. It loads, chunks, and stores document data using vector stores, and utilizes LLMs to respond to queries based on the contextual data.

## Structure

* **`app/ingestion/`**: Contains code for loading documents and chunking.
* **`app/retrieval/`**: Logic to retrieve relevant document chunks given a query.
* **`app/vector_store/`**: Configuration and connections for the vector database.
* **`app/llm/`**: Code interacting with language models.
* **`app/rag_pipeline/`**: The main orchestration combining components for the end-to-end RAG pipeline.

## Getting Started

1. Place your data (e.g. PDF files) in the `app/data/` or `data/` directory.
2. Ensure you have the required dependencies installed (refer to `requirements.txt`).
   ```bash
   pip install -r requirements.txt
   ```
3. Set your environment variables in `.env`.
4. Run the main processing script:
   ```bash
   streamlit run app/app.py
   ```
I implemented:

Multi-query retrieval

Hybrid search (BM25 + vector)

MMR retrieval

Deduplication

Cross-encoder reranking

Token-safe context building


What Your Current Pipeline Does

Your flow currently becomes:

User query
    ↓
Generate multiple queries
    ↓
Hybrid retriever (Vector MMR + BM25)
    ↓
Retrieve docs for each query
    ↓
Merge results
    ↓
Deduplicate
    ↓
Rerank documents
    ↓
Token-safe context builder
    ↓
LLM

This is already a strong RAG architecture.


Your RAG Architecture (Overall)

Your pipeline flow currently is:

User Query
     │
Multi Query Generation (LLM)
     │
Hybrid Retrieval
  ├─ Vector Search (Qdrant + embeddings)
  └─ Keyword Search (BM25)
     │
Deduplication
     │
Cross Encoder Reranking
     │
Token Limited Context Builder
     │
LLM Generation
     │
Final Answer

This is very close to modern RAG pipelines used in industry.
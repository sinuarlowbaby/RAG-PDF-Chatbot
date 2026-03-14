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
   python app/main.py
   ```

import os
import dotenv

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from rag_pipeline.rag_pipeline import rag_pipeline




dotenv.load_dotenv()

client = QdrantClient(url="http://localhost:6333/")

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    chunk_size=100)

user_query = "What is python?"

rag_pipeline(client, embedding_model,user_query)
print("✅rag pipeline done")




import os
import dotenv

from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from pipeline.ingest_pipeline import ingest_pipeline
from pipeline.query_pipeline import query_pipeline
import streamlit as st

dotenv.load_dotenv()
client = QdrantClient(url="http://localhost:6333")

user_query = "features of python"

vector_store,documents = ingest_pipeline(client)
response = query_pipeline(vector_store,user_query,documents,client)

print("✅rag pipeline done")

print(f"""➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️\n 
        {response} \n
         ➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️""")

client.close()

import os
import dotenv

# from langchain_community.cache import RedisCache
# from langchain.globals import set_llm_cache
from retrival.hybrid_document_retrival import initialize_retrievers
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from pipeline.ingest_pipeline import ingest_pipeline
from pipeline.query_pipeline import query_pipeline
import streamlit as st
import openai

dotenv.load_dotenv()
client = QdrantClient(url="http://localhost:6333")

# user_query = "features of python"

# redis_client = redis.Redis(host="localhost", port=6379, db=0)
# set_llm_cache(RedisCache(redis_client))
# print("✅ Redis cache initialized")

vector_store,documents = ingest_pipeline(client)
hybrid_retriver = initialize_retrievers(vector_store,documents,20)


while True:
        user_query = input("Enter your query ➡️ : ").strip().lower()
        if user_query == "exit":
            break
        response_generator= query_pipeline(vector_store,user_query,documents,client,hybrid_retriver)

        full_response = ''
        try:
            for chunk in response_generator:
                print(chunk, end='', flush=True)
                full_response += chunk

        except openai.APITimeoutError:
            print("\n❌ [Timeout Error]: OpenAI took too long to respond. Please try again.")
        except openai.APIStatusError as e:
            print(f"\n❌ [API Error]: OpenAI returned an error status: {e.status_code}")
        except Exception as e:
            print(f"\n❌ [Unexpected Error]: Something broke during the stream: {e}")

        finally:
            print("\n➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️")
        


        print("✅rag pipeline done")
        
client.close()

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
while True:
        user_query = input("Enter your query ➡️ : ")
        if user_query == "exit":
            break
        response_generator = query_pipeline(vector_store,user_query,documents,client)

        print("✅rag pipeline done")
        complete_response =""
        print("➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️\n ")
        
        try:
                for chunk in response_generator:
                        print(chunk, end='', flush=True)
                        complete_response += chunk

        except openai.APIConnectionError:
            print("\n❌ [Network Error]: Connection to OpenAI dropped. Please check your internet.")
        except openai.APITimeoutError:
            print("\n❌ [Timeout Error]: OpenAI took too long to respond. Please try again.")
        except openai.APIStatusError as e:
            print(f"\n❌ [API Error]: OpenAI returned an error status: {e.status_code}")
        except Exception as e:
            print(f"\n❌ [Unexpected Error]: Something broke during the stream: {e}")

        finally:
            print("\n➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️➡️")
        
        client.close()

import os
import dotenv

from retrival.hybrid_document_retrival import initialize_retrievers
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from pipeline.ingest_pipeline import ingest_pipeline
from pipeline.query_pipeline import query_pipeline
import openai

dotenv.load_dotenv()
client = QdrantClient(url="http://localhost:6333")

vector_store,documents = ingest_pipeline(client)
hybrid_retriver = initialize_retrievers(vector_store,documents,20)


try:
    while True:
        user_query = input("\nEnter your query ➡️ : ").strip().lower()
        if user_query in ["exit", "quit"]:
            print("\n👋 Exiting RAG pipeline...")
            break
        
        response_generator = query_pipeline(vector_store, user_query, documents, client, hybrid_retriver)

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

        print("✅ rag pipeline done")

except KeyboardInterrupt:
    print("\n👋 Exiting due to keyboard interrupt...")
finally:
    client.close()
    print("✅ System resources cleaned up. Goodbye!")

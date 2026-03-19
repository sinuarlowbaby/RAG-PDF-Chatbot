from langchain_openai import OpenAIEmbeddings
import dotenv
import os

dotenv.load_dotenv()

def user_query_embedding(user_query):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        chunk_size=100
        )

    embedding = embedding_model.embed_query(user_query)
    return embedding
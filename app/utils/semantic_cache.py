import redis
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import datetime


redis_client = redis.Redis(
    host="localhost", 
    port=6379, 
    db=0,
    decode_responses=True
    )



# def cosine_similarity(vec1, vec2):
#     #measure similarity between two vectors

#     vec1 = np.array(vec1)
#     vec2 = np.array(vec2)

#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def semantic_cache_match(user_query_embedding,treshold=0.8):
    keys = redis_client.keys("semantic_cache:*")
    for key in keys:
        data = json.loads(redis_client.get(key))
        cached_embedding = data["embedding"]
        cached_context = data["context"]    # context created from pdf
        cached_response = data["response"]  # llm generated response for the context

        similarity_score = cosine_similarity(
            [cached_embedding],
            [user_query_embedding]
            )[0][0]

        if similarity_score > treshold:
            return cached_response
    
    return None


def store_semantic_cache(user_query,query_embedding,context,response):
    
    if hasattr(query_embedding,'tolist'):
        query_embedding = query_embedding.tolist()
    if hasattr(context,'tolist'):
        context = context.tolist()
    if hasattr(response,'tolist'):
        response = response.tolist()

    data={
        "user_query":user_query,
        "embedding":query_embedding,
        "context":context,
        "response":response,
        "created_at":datetime.datetime.now().isoformat()
    }

    key = f"semantic_cache:{uuid.uuid4()}"
    
    redis_client.set(
        key,
        json.dumps(data),
        ex=3600
    )

    return True
    
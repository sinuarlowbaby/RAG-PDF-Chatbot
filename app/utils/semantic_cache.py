import redis.asyncio as redis
import json
from sklearn.metrics.pairwise import cosine_similarity
import uuid
import datetime


redis_client = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,
    socket_timeout=0.1,
    socket_connect_timeout=0.1,
    retry_on_timeout=False,
)


async def redis_available() -> bool:
    try:
        return await redis_client.ping()
    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError, redis.exceptions.RedisError):
        return False


async def semantic_cache_match(user_query_embedding, threshold=0.7):
    if not await redis_available():
        return None

    try:
        async for key in redis_client.scan_iter("semantic_cache:*"):
            raw = await redis_client.get(key)
            if raw is None:
                continue
            data = json.loads(raw)
            cached_embedding = data["embedding"]
            cached_context = data["context"]    # context created from pdf
            cached_response = data["response"]  # llm generated response for the context

            similarity_score = cosine_similarity(
                [cached_embedding],
                [user_query_embedding]
            )[0][0]

            if similarity_score > threshold:
                return cached_response
        return None
    except redis.exceptions.RedisError:
        return None


async def store_semantic_cache(user_query,query_embedding,context,response):
    if not await redis_available():
        return False

    try:
        if hasattr(query_embedding, 'tolist'):
            query_embedding = query_embedding.tolist()
        if hasattr(context, 'tolist'):
            context = context.tolist()
        if hasattr(response, 'tolist'):
            response = response.tolist()

        data = {
            "user_query": user_query,
            "embedding": query_embedding,
            "context": context,
            "response": response,
            "created_at": datetime.datetime.now().isoformat()
        }

        key = f"semantic_cache:{uuid.uuid4()}"

        await redis_client.set(
            key,
            json.dumps(data),
            ex=3600
        )

        return True
    except redis.exceptions.RedisError:
        return False
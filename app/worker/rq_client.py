import redis
from rq import Queue
import os
from dotenv import load_dotenv

load_dotenv()


# Configuration - using defaults for Docker (localhost:6379)
redis_conn = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379))
)


upload_q = Queue('upload', connection=redis_conn)
chat_q = Queue('chat', connection=redis_conn)

print("RQ Client initialized")

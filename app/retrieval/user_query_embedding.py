from openai import AsyncOpenAI
import dotenv
import os

dotenv.load_dotenv()

async_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def user_query_embedding(user_query: str) -> list:
    response = await async_client.embeddings.create(
        model="text-embedding-3-small",
        input=user_query,
    )
    return response.data[0].embedding
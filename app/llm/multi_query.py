from openai import AsyncOpenAI
import asyncio
import dotenv
import os
import json

dotenv.load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def generate_queries(user_query, n_queries=4):
    SYSTEM_PROMPT = f"""
        You are a search query generator.

        Generate {n_queries} different search queries that could retrieve
        relevant documents from a vector database.

        Rules:
            - Queries must have different wording
            - Maintain the same meaning
            - Return ONLY a JSON list

        Example:
            ["query1", "query2", "query3", "query4"]
        """

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ],
        temperature=0.3
    )
    try:
        queries=json.loads(response.choices[0].message.content)
        if not isinstance (queries,list):
            raise ValueError("Expected a JSON list from LLM")
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        print(f"⚠️ Query generation parse failed, falling back to original query: {e}")
        queries=[user_query]

    return queries

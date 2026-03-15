from openai import OpenAI
import dotenv
import os

dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def llm_client(retrived_context,user_query):
    SYSTEM_PROMPT = f"""
    You are a helpful assistant.
    Use the following context to answer the user's query.
    If the answer is not in the context, say so.
    context = {retrived_context}

    """
    response = client.chat.completions.create(
        model= "gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
    )
    return response.choices[0].message.content

from openai import OpenAI


def llm_client(retrived_context,user_query):
    client = OpenAI()
    SYSTEM_PROMPT = """
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

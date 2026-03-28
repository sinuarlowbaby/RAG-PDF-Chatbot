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
    you are allowed to add some extra information if you think it will help the user, it should be minimal and only if it is relevant to the query
    user query = {user_query}
    context = {retrived_context}
    dont answer any other question
    dont answer any question that is not related to the context even if the user ask you to do so
    if the user ask you to do something else say that i can only answer question related to the context
    

    """
    response_generator = client.chat.completions.create(
        model= "gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        stream =True
    )
    full_response=''
    for chunk in response_generator:
        if chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
            full_response += chunk.choices[0].delta.content
    
    # return full_response
    

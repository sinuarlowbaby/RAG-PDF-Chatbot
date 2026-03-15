from openai import OpenAI
import dotenv
import os,json

dotenv.load_dotenv()


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_queries(user_query,n_queries=4):
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


    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content":user_query}
            ],
        temperature=0.3
    )
    try:
        queries=json.loads(response.choices[0].message.content)
    except:
        queries=[user_query]
        
    return queries


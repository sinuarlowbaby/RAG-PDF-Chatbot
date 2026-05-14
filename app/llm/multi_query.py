from openai import OpenAI
import dotenv
import os,json
from langsmith.wrappers import wrap_openai
from langsmith import traceable

dotenv.load_dotenv()


client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

@traceable(run_type="llm", name="Generate_Multiple_Queries")
def generate_queries(user_query,n_queries=4):
    SYSTEM_PROMPT = f"""
        You are an expert query expansion engine for a semantic vector search system.

        Your task is to generate {n_queries} high-quality search queries that maximize document retrieval recall.

        Guidelines:
        - Each query must represent a DIFFERENT retrieval strategy:
            1. Keyword-focused query (short, dense terms)
            2. Natural language question
            3. Semantic paraphrase
            4. Expanded context query (add related concepts)

        - Preserve the original intent exactly
        - Use different vocabulary and structure
        - Avoid repeating phrases
        - Keep queries concise but meaningful
        - Avoid overly generic queries
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content":user_query}
            ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


from openai import OpenAI
import dotenv
import os,json
from langsmith.wrappers import wrap_openai
from langsmith import traceable

dotenv.load_dotenv()


client = wrap_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

@traceable(run_type="llm", name="Generate_Multiple_Queries")
def generate_queries(user_query):
    query_len = len(user_query.split())
    if query_len < 5:
        n_queries=2
    else:
        n_queries=4

    SYSTEM_PROMPT = f"""
      You are a retrieval query optimization engine.

      Generate {n_queries} semantically distinct search queries for a hybrid RAG retrieval system.

      Goals:
      - maximize retrieval recall
      - preserve original intent
      - improve semantic coverage
      - avoid semantic drift
      - avoid redundant wording

      Rules:
      - keep queries concise
      - each query should target a different retrieval angle
      - preserve technical terms
      - do NOT explain anything
      - do NOT number the output
      - return ONLY a valid JSON array of strings

      Good query types:
      - keyword-focused
      - semantic paraphrase
      - natural language variation
      - context-expanded variation (ONLY if useful)

      Bad behavior:
      - overly broad queries
      - unrelated concepts
      - generic filler text
      - conversational explanations

      Example output:
      [
        "python programming language",
        "what is python used for"
      ]
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":SYSTEM_PROMPT},
            {"role":"user", "content":user_query}
            ],
        temperature=0.3
    )
    queries = json.loads(response.choices[0].message.content.strip())
    return queries

    # output = response.choices[0].message.content.strip()
    # return output


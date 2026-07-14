import json
import logging
import os

from dotenv import load_dotenv
from groq import Groq
from langsmith import traceable

load_dotenv()

logger = logging.getLogger(__name__)

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


@traceable(run_type="llm", name="Generate_Multiple_Queries", metadata={"model": "llama-3.3-70b-versatile"})
def generate_queries(user_query: str) -> list[str]:
    """Generate semantically diverse search queries from the user's original question.

    Short queries (<5 words) get 2 variants; longer ones get 4, to balance
    retrieval recall against latency.

    Args:
        user_query: The original question from the user.

    Returns:
        A list of query strings. Falls back to [user_query] on parse error.
    """
    query_len = len(user_query.split())
    n_queries = 2 if query_len < 5 else 4

    system_prompt = f"""
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

    response = _groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.3,
    )

    try:
        queries = json.loads(response.choices[0].message.content.strip())
    except json.JSONDecodeError:
        logger.warning("Multi-query parse failed — falling back to original query")
        queries = [user_query]

    logger.debug(f"Generated {len(queries)} query variant(s) for: {user_query[:50]!r}")
    return queries

from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask",
        examples=["What is python?"]
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="LLM temperature"
    )
    use_answer_cache: bool = Field(
        True,
        description="Enable/disable semantic answer cache"
    )
    use_multi_query_cache: bool = Field(
        True,
        description="Enable/disable multi-query expansion cache"
    )

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    timestamp: str
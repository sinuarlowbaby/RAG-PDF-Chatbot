from pydantic import BaseModel,Field

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

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    timestamp: str
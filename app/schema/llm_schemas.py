from pydantic import BaseModel,Field

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="The question to ask",
        examples=["What is python?"]
    )

class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    timestamp: str
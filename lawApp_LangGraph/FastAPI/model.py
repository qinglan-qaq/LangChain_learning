from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    final_answer: str
    messages: list
    
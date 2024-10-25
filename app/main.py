from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from .preprocess import preprocess
from .get_docs import get_docs

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


class DocumentResponse(BaseModel):
    documents: List[str]
    similarity: List[Union[float, int]]


@app.post("/search", response_model=DocumentResponse)
async def search(query_request: QueryRequest):
    query = query_request.query
    query_embedding = preprocess(query)
    top_documents, similarity = get_docs(query_embedding)
    return {"documents": top_documents, "similarity": similarity}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

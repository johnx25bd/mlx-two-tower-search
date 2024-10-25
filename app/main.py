from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union

from .preprocess import preprocess
from .get_docs import get_docs

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

class DocumentResponse(BaseModel):
    rel_docs: List[str]
    rel_docs_sim: List[Union[float, int]]
    irel_docs: List[str]
    irel_docs_sim: List[Union[float, int]]

@app.post("/search", response_model=DocumentResponse)
async def search(query_request: QueryRequest):
    query = query_request.query
    query_embedding = preprocess(query)
    rel_docs, rel_docs_sim, irel_docs, irel_docs_sim = get_docs(query_embedding)
    return {
        "rel_docs": rel_docs,
        "rel_docs_sim": rel_docs_sim,
        "irel_docs": irel_docs,
        "irel_docs_sim": irel_docs_sim,
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

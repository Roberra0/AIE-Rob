from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# Import the clean function from rag_pipeline
from rag.rag_pipeline import process_query

app = FastAPI(title="ApartmentSearch.ai API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    query: str
    response: str
    context_count: int
    similarity_scores: list
    context: list

@app.get("/")
async def root():
    return {"message": "ApartmentSearch.ai API is running!"}

# Process the query from the API, then run our rag pipeline
@app.post("/query", response_model=QueryResponse)
async def process_query_endpoint(request: QueryRequest):
    """Process a natural language query about apartments using RAG pipeline"""
    try:
        result = process_query(request.query)
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            context_count=result["context_count"],
            similarity_scores=result["similarity_scores"],
            context=result["context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
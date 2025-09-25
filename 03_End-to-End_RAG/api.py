from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag.rag_pipeline import main as run_rag_pipeline

app = FastAPI(title="ApartmentSearch.ai API", version="1.0.0")

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
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

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query about apartments using RAG pipeline
    """
    try:
        # Run the RAG pipeline with the user's query
        result = await run_rag_query(request.query)
        
        return QueryResponse(
            query=result["query"],
            response=result["response"],
            context_count=result["context_count"],
            similarity_scores=result["similarity_scores"],
            context=result["context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

async def run_rag_query(user_query: str):
    """
    Run the RAG pipeline with a specific query
    """
    # Import here to avoid circular imports
    from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
    from aimakerspace.vectordatabase import VectorDatabase
    from aimakerspace.openai_utils.chatmodel import ChatOpenAI
    from rag.rag_pipeline import RetrievalAugmentedQAPipeline
    
    # Load and split documents
    loader = TextFileLoader("data/rental_listings.pdf")
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separator="\n"
    )
    split_documents = text_splitter.split_documents(documents)
    
    # Build vector database
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(split_documents)
    
    # Initialize RAG pipeline
    chat_openai = ChatOpenAI()
    rag_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        response_style="detailed",
        include_scores=True
    )
    
    # Process the query
    result = rag_pipeline.run(user_query)
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

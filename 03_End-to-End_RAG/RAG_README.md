# RAG Pipeline - Rental Listings

A complete Retrieval-Augmented Generation (RAG) pipeline for querying rental apartment listings from PDF documents.

## Features

- **PDF Processing**: Extracts text from PDFs using pdfplumber with smart column detection
- **Vector Search**: Uses OpenAI embeddings for semantic search
- **RAG Pipeline**: Combines vector search with LLM for intelligent responses
- **Multi-format Support**: Handles both .txt and .pdf files
- **Smart Chunking**: Splits documents into optimal chunks for retrieval

## Quick Start

1. **Set up environment**:
   ```bash
   uv sync
   ```

2. **Add your OpenAI API key** to `.env`:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the pipeline**:
   ```bash
   uv run python rag_pipeline.py
   ```

## File Structure

```
├── rag_pipeline.py          # Main RAG pipeline
├── config.py               # Configuration settings
├── test_queries.py         # Test different query types
├── aimakerspace/           # Custom utilities
│   ├── text_utils.py       # PDF/text processing
│   ├── vectordatabase.py   # Vector search
│   └── openai_utils/       # OpenAI integration
└── data/                   # Document storage
    └── Rental_Listings.pdf # Sample rental data
```

## Key Components

### TextFileLoader
- Loads .txt and .pdf files
- Smart column detection for multi-column PDFs
- Handles font warnings automatically

### RetrievalAugmentedQAPipeline
- Combines vector search with LLM
- Configurable response styles and lengths
- Returns structured results with context sources

### VectorDatabase
- Uses OpenAI text-embedding-3-small
- Cosine similarity search
- Async support for building indexes

## Example Usage

```python
from rag_pipeline import RetrievalAugmentedQAPipeline, vector_db
from aimakerspace.openai_utils.chatmodel import ChatOpenAI

# Initialize pipeline
chat_openai = ChatOpenAI()
pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    response_style="detailed",
    include_scores=True
)

# Query the system
result = pipeline.run_pipeline(
    "What apartments are available for under $3000?",
    k=3,
    response_length="comprehensive"
)

print(result['response'])
```

## Configuration

Edit `config.py` to customize:
- Document paths
- Chunk sizes
- Response styles
- Model settings

## Testing

Run test queries to validate the system:
```bash
uv run python test_queries.py
```

## Next Steps

This RAG pipeline is ready for frontend integration. The main components can be imported and used in web applications or APIs.

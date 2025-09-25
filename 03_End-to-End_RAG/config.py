"""
Configuration file for RAG Pipeline
"""
import os

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Document Processing
DEFAULT_DOCUMENT_PATH = "data/Rental_Listings.pdf"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# RAG Pipeline Settings
DEFAULT_K = 3  # Number of context chunks to retrieve
DEFAULT_RESPONSE_STYLE = "detailed"
DEFAULT_RESPONSE_LENGTH = "comprehensive"
INCLUDE_SIMILARITY_SCORES = True

# Vector Database Settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

# Chat Model Settings
CHAT_MODEL = "gpt-4o-mini"

# Output Settings
PRINT_CONTEXT_LENGTH = 300  # Characters to show in context preview

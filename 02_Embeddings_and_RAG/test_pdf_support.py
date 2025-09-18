#!/usr/bin/env python3
"""
Test script to demonstrate PDF support in the RAG pipeline.
This shows how to use the updated TextFileLoader with PDF files.
"""

from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio

def test_pdf_loading():
    """Test loading a PDF file and processing it through the RAG pipeline."""
    
    # Example usage with a PDF file
    # Replace 'your_pdf_file.pdf' with the actual path to your PDF
    pdf_path = "data/your_pdf_file.pdf"  # Update this path
    
    print("Testing PDF loading...")
    
    # Load PDF
    loader = TextFileLoader(pdf_path)
    documents = loader.load_documents()
    
    print(f"Loaded {len(documents)} document(s) from PDF")
    print(f"First document length: {len(documents[0])} characters")
    print(f"First 200 characters: {documents[0][:200]}...")
    
    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_texts(documents)
    
    print(f"Split into {len(split_documents)} chunks")
    print(f"First chunk: {split_documents[0][:200]}...")
    
    return split_documents

def test_pdf_with_rag():
    """Test the complete RAG pipeline with PDF content."""
    
    # Load and process PDF
    split_documents = test_pdf_loading()
    
    # Create vector database
    print("\nCreating vector database...")
    vector_db = VectorDatabase()
    
    # Build the vector database (this requires API key to be set)
    try:
        vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))
        print("Vector database created successfully!")
        
        # Test search
        print("\nTesting search...")
        results = vector_db.search_by_text("What is the main topic?", k=3)
        print(f"Found {len(results)} relevant chunks")
        
        for i, (chunk, score) in enumerate(results):
            print(f"\nResult {i+1} (score: {score:.3f}):")
            print(chunk[:200] + "...")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your OpenAI API key is set!")

if __name__ == "__main__":
    print("PDF Support Test")
    print("================")
    
    # Test basic PDF loading
    test_pdf_loading()
    
    # Uncomment the line below to test the full RAG pipeline
    # test_pdf_with_rag()

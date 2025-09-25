"""
Test script to validate RAG pipeline with different query types
"""
import sys
sys.path.append('.')

from rag_pipeline import vector_db, RetrievalAugmentedQAPipeline
from aimakerspace.openai_utils.chatmodel import ChatOpenAI

def test_query(query, description):
    """Test a single query and display results"""
    print(f"\n{'='*80}")
    print(f"TEST: {description}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")
    
    # Initialize pipeline
    chat_openai = ChatOpenAI()
    rag_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        response_style="detailed",
        include_scores=True
    )
    
    # Run query
    result = rag_pipeline.run_pipeline(query, k=3)
    
    # Display results
    print(f"💬 Response: {result['response']}")
    print(f"📊 Context Count: {result['context_count']}")
    print(f"📈 Top Similarity Score: {result['similarity_scores'][0] if result['similarity_scores'] else 'N/A'}")

def main():
    """Run various test queries"""
    test_queries = [
        ("Are there any 2 bedroom apartments?", "Specific bedroom count"),
        ("What's the cheapest apartment available?", "Price-based query"),
        ("Show me apartments near UC Berkeley", "Location-based query"),
        ("Are there any pet-friendly apartments?", "Feature-based query"),
        ("What amenities are included?", "General information query"),
        ("How much does a 1 bedroom cost?", "Cost inquiry"),
        ("Are there any apartments with parking?", "Specific amenity query"),
    ]
    
    for query, description in test_queries:
        try:
            test_query(query, description)
        except Exception as e:
            print(f"❌ Error with query '{query}': {e}")
    
    print(f"\n{'='*80}")
    print("✅ All tests completed!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

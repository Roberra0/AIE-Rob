import sys
import os
# Add parent directory to path so we can import aimakerspace
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from numpy.linalg import norm
import openai
from getpass import getpass
#The aimakerspace utilities provide production-ready, optimized components that would take significant time to build from scratch. Even though youre using the OpenAI API directly, these utilities give you
from aimakerspace.openai_utils.embedding import EmbeddingModel
from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio
import nest_asyncio
from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    AssistantRolePrompt,
)
from aimakerspace.openai_utils.chatmodel import ChatOpenAI

# Helps determine how close two vectors are to each other  
def cosine_similarity(vec_1, vec_2):
  return np.dot(vec_1, vec_2) / (norm(vec_1) * norm(vec_2))

# Set up the OpenAI API key from the .env file
openai.api_key = os.getenv('OPENAI_API_KEY')


# Set up the embedding model
embedding_model = EmbeddingModel()


### TEST
# Will two similar sentences have a high cosine similarity?
#puppy_sentence = "I have a cute puppy"
#dog_sentence = "That is my puppy"
#puppy_vector = embedding_model.get_embedding(puppy_sentence)
#dog_vector = embedding_model.get_embedding(dog_sentence)

#print("Similarity between puppy and dog vectors:", cosine_similarity(puppy_vector, dog_vector))

# Will a vector calculation result be close to a target vector?
#king_vector = np.array(embedding_model.get_embedding("King")) # Use np.array to convert the embedding to a numpy array which we can do calculations on
#man_vector = np.array(embedding_model.get_embedding("man"))
#woman_vector = np.array(embedding_model.get_embedding("woman"))

#vector_calculation_result = king_vector - man_vector + woman_vector
#queen_vector = np.array(embedding_model.get_embedding("Queen"))

#print("Similarity between vector calculation result and queen vector:", cosine_similarity(vector_calculation_result, queen_vector))
### TEST END



### Load the file
text_loader = TextFileLoader("data/rental_listings.pdf") # TextFileLoader is a class that loads text files
documents = text_loader.load_documents() # Load the documents from the text file
len(documents)
# print("First document 10 chars:", documents[0][:10]) # For the first document [0], print the first 100 characters :100 

### Split the documents into chunks
text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)
# print("# of chunks:", len(split_documents))
# print("First chunk:", split_documents[0:1])


### Build the vector database, 
# DB has a default embedding model (OpenAI: text-embedding-3-small), Embedding dimension is 1536, context window is 8191
vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

# TEST Query the vector database test
# Embeds our query witht he same ebedding model, and loops through every vector in the DB and calculates cosine similarity to return top k closest vectors
# query = "Are there any 2 bedroom 1bathroom apartments that are ~800 sqft?"
# result = vector_db.search_by_text(query, k=3)
# print(query, "\n", result)


### PROMPTS
# 1. You start with a system message that outlines how the LLM should respond, what kind of behaviours you can expect from it, and more
# 2. Then, you can provide a few examples in the form of "assistant"/"user" pairs
# 3. Then, you prompt the model with the true "user" message.
# The below is constructed in a way to help dynamically assign roles to the model


### TEST Chat
# chat_openai = ChatOpenAI() 
# user_prompt_template = "{content}"
# system_prompt_template = (
#     "You are an expert in {expertise}, you always answer in a kind way." 
# )
# user_role_prompt = UserRolePrompt(user_prompt_template) 
# system_role_prompt = SystemRolePrompt(system_prompt_template) 

# messages = [
#     system_role_prompt.create_message(expertise="brief simple explanations to high school students"),
#     user_role_prompt.create_message(
#         content="What is the best way to write a loop?"
#     ),
# ]

# response = chat_openai.run(messages)
# print(response)


RAG_SYSTEM_TEMPLATE = """You are a knowledgeable real estate agent that answers questions based strictly on provided context, you are aiding user in finding an apartment.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response.
- Be kind and warm, you're trying to help the user find their dream home"""

RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

rag_system_prompt = SystemRolePrompt(
    RAG_SYSTEM_TEMPLATE,
    strict=True,
    defaults={
        "response_style": "concise",
        "response_length": "brief"
    }
)
rag_user_prompt = UserRolePrompt(
    RAG_USER_TEMPLATE,
    strict=True,
    defaults={
        "context_count": "",
        "similarity_scores": ""
    }
)

### RAG Pipeline, a class that takes in a LLM, a vector database, and a response style, and returns a response
class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> dict:
        # Retrieve relevant contexts
        context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
        
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)

        return {
            "query": user_query,
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }

## MAIN FUNCTION TO RUN THE RAG PIPELINE WITH A SAMPLE QUERY, ONLY RUNS WHEN THE SCRIPT IS RUN DIRECTLY, NOT WHEN IMPORTED AS A MODULE
def main():
    """Main function to run the RAG pipeline"""
    # Initialize components
    chat_openai = ChatOpenAI() 
    rag_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        response_style="detailed",
        include_scores=True
    )

    # Example query
    query = "What apartments are available that are $4,000?"
    
    # Run the pipeline
    result = rag_pipeline.run_pipeline(
        query,
        k=3,
        response_length="comprehensive", 
        response_style="detailed",
        include_warnings=True,
        confidence_required=True
    )

    # Display results
    print(f"🔍 QUERY: {result.get('query', 'No query found')}")
    print(f"\n💬 Response: {result['response']}")
    print(f"\n📊 Context Count: {result['context_count']}")
    print(f"📈 Similarity Scores: {result['similarity_scores']}")

    print(f"\n{'='*80}")
    print("CONTEXT SOURCES:")
    print(f"{'='*80}")

    for i, (context, score) in enumerate(result['context'], 1):
        print(f"\n📊 SIMILARITY SCORE: {score:.3f}")
        print(f"📄 SOURCE {i}:")
        print(f"{'─'*60}")
        print(f"{context[:800]}...")  # First 800 chars to see more of the chunk
        print(f"{'─'*60}")
        print()  # Extra blank line for separation

### API Functions to create RAG pipeline and process query from API
def create_rag_pipeline():
    """Create and return a configured RAG pipeline instance"""
    chat_openai = ChatOpenAI() 
    rag_pipeline = RetrievalAugmentedQAPipeline(
        vector_db_retriever=vector_db,
        llm=chat_openai,
        response_style="detailed",
        include_scores=True
    )
    return rag_pipeline

def process_query(query: str, k: int = 3):
    """Process a single query using the RAG pipeline - can be imported by API"""
    rag_pipeline = create_rag_pipeline()
    
    result = rag_pipeline.run_pipeline(
        query,
        k=k,
        response_length="comprehensive", 
        response_style="detailed",
        include_warnings=True,
        confidence_required=True
    )
    return result


if __name__ == "__main__": # Calls the main function only when the script is run directly, not when imported as a module
    main()
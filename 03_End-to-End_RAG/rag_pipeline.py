import numpy as np
from numpy.linalg import norm
import os
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
with open('.env', 'r') as f:
    line = f.read().strip()
    if line.startswith('OPENAI_API_KEY='):
        openai.api_key = line.split('=', 1)[1]


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
text_loader = TextFileLoader("data/random_book.txt") # TextFileLoader is a class that loads text files
documents = text_loader.load_documents() # Load the documents from the text file
len(documents)
print("First document 10 chars:", documents[0][:10]) # For the first document [0], print the first 100 characters :100 

### Split the documents into chunks
text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)
# print("# of chunks:", len(split_documents))
# print("First chunk:", split_documents[0:1])


### Build the vector database, 
# DB has a default embedding model (OpenAI: text-embedding-3-small), Embedding dimension is 1536, context window is 8191
vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))

# Query the vector database
# Embeds our query witht he same ebedding model, and loops through every vector in the DB and calculates cosine similarity to return top k closest vectors
query = "What is it important to know in any research work?"
result = vector_db.search_by_text("What is it important to know in any research work?", k=3)
# print(query, "\n", result)


### PROMPTS
# 1. You start with a system message that outlines how the LLM should respond, what kind of behaviours you can expect from it, and more
# 2. Then, you can provide a few examples in the form of "assistant"/"user" pairs
# 3. Then, you prompt the model with the true "user" message.
# The below is constructed in a way to help dynamically assign roles to the model


### TEST Chat
chat_openai = ChatOpenAI() 
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


RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

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
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }


### TEST RAG Pipeline
rag_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    response_style="detailed",
    include_scores=True
)

result = rag_pipeline.run_pipeline(
    "What is is something important to know in any research work?",
    k=3,
    response_length="comprehensive", 
    include_warnings=True,
    confidence_required=True
)

print(f"Response: {result['response']}")
print(f"\nContext Count: {result['context_count']}")
print(f"Similarity Scores: {result['similarity_scores']}")
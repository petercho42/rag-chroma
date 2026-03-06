import ollama
import chromadb

# 1. SETUP
CHROMA_HOST = 'localhost'
CHROMA_PORT = 8001
# OLLAMA_MODEL = "deepseek-r1:32b" # Your heavy hitter
OLLAMA_MODEL = "qwen2.5:32b"

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
ollama_client = ollama.Client(host='http://localhost:11434')

collection = chroma_client.get_collection(name="personal_knowledge_base")

def generate_rag_answer(question):
    # STEP A: Embed the question
    query_embed = ollama_client.embeddings(model="nomic-embed-text", prompt=question)['embedding']

    # STEP B: Retrieve Top 3 chunks
    results = collection.query(query_embeddings=[query_embed], n_results=10)
    context = "\n\n".join(results['documents'][0])

    # STEP C: Create the "Augmented" Prompt
    # We give the model a specific role and strict rules.
    prompt = f"""
    You are a helpful assistant. Answer the user's question ONLY using the provided context from a research paper. 
    If the answer is not in the context, say you don't know.

    CONTEXT FROM PAPER:
    {context}

    USER QUESTION:
    {question}

    ANSWER:
    """

    # STEP D: Generate with the 3090
    print(f"\n--- Thinking with {OLLAMA_MODEL}... ---")
    response = ollama_client.generate(model=OLLAMA_MODEL, prompt=prompt)
    
    print("\n--- FINAL ANSWER ---")
    print(response['response'])

if __name__ == "__main__":
    user_q = "What is the main benefit of the Transformer architecture?"
    generate_rag_answer(user_q)
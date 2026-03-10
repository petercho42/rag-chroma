import ollama
import chromadb

# 1. SETUP
CHROMA_HOST = 'localhost'
CHROMA_PORT = 8001
OLLAMA_MODEL = "deepseek-r1:32b" # Your heavy hitter
# OLLAMA_MODEL = "qwen2.5:32b"

chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
ollama_client = ollama.Client(host='http://localhost:11434')

collection = chroma_client.get_collection(name="personal_knowledge_base")

def generate_targeted_answer(question, target_pages=[0, 1, 10, 11]):
    query_embed = ollama_client.embeddings(model="nomic-embed-text", prompt=question)['embedding']

    # This is the "Sniper" part:
    # We tell Chroma only to return chunks where the 'page' is in our list
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=5,
        where={"page": {"$in": target_pages}} 
    )

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
    user_q = "According to the Introduction of this paper, what were the main limitations of previous models?"
    generate_targeted_answer(user_q)
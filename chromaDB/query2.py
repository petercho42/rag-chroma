import ollama
import chromadb

# 1. Connect to the 3090 services
chroma_client = chromadb.HttpClient(host='localhost', port=8001)
ollama_client = ollama.Client(host='http://localhost:11434')

collection = chroma_client.get_collection(name="personal_knowledge_base")

def ask_question(query):
    # STEP A: Turn your question into math
    query_embed = ollama_client.embeddings(model="nomic-embed-text", prompt=query)['embedding']

    # STEP B: Search the 3090 for the top 3 most relevant matches
    results = collection.query(
        query_embeddings=[query_embed],
        n_results=3
    )

    # STEP C: Print the findings
    print(f"\n--- Results for: '{query}' ---")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n[Match {i+1}]:")
        print(doc)

if __name__ == "__main__":
    # Test it with a question specifically about the PDF you loaded
    ask_question("What is the main benefit of the Transformer architecture?")
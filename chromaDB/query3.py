import ollama
import chromadb

def pdf_search_tool(query: str):
    """Searches the 'Attention is All You Need' PDF for technical answers."""
    client = chromadb.HttpClient(host='localhost', port=8001)
    collection = client.get_collection(name="personal_knowledge_base")
    
    # 1. Embed the query
    embed = ollama.embeddings(model="nomic-embed-text", prompt=query)['embedding']
    
    # 2. Get Top 5 matches
    results = collection.query(query_embeddings=[embed], n_results=5)
    return "\n\n".join(results['documents'][0])

# Simple calculator tool (LLMs are bad at math, so we give them a tool)
def calculator_tool(expression: str):
    """Executes basic math expressions like '512 * 2'."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"
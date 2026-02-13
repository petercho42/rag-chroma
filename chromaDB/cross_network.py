import chromadb
from ollama import Client
from augment_prompt import augment_prompt
from database import collection

# 1. Connect to your 3090 (Replace with your server's local IP)
# Make sure you did the 'OLLAMA_HOST=0.0.0.0' step on the server!
server_3090 = Client(host="http://szuni:11434")

user_query = "What specific hardware chip is in the iPhone 16?"
# user_query = "What is the weather like in Seoul in the spring?"

# Retrieve top 2 matches
results = collection.query(query_texts=[user_query], n_results=2)
chunks = results["documents"][0]  # This is a list of the text results

# 3. Augment
final_prompt = augment_prompt(user_query, chunks)

# 4. Generate
print("Thinking... (Consulting the 3090)")
response = server_3090.generate(model="qwen3:30b", prompt=final_prompt)

print("-" * 30)
print(f"QUERY: {user_query}")
print(f"RESULT: {response['response']}")

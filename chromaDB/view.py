from database import collection


# Retrieves all documents and metadatas (excluding embeddings for readability)
data = collection.get()

print(f"Total documents: {len(data['ids'])}")
for i in range(len(data["ids"])):
    print(f"ID: {data['ids'][i]}")
    print(f"Metadata: {data['metadatas'][i]}")
    print(f"Document: {data['documents'][i][:100]}...")  # Just first 100 chars
    print("-" * 20)

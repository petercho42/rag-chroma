import chromadb

# 1. Initialize the client (This creates a local database in memory)
client = chromadb.Client()

# 2. Create a collection (Think of this as your "Table")
# We'll call it "my_rag_docs"
collection = client.create_collection(name="my_rag_docs")

# 3. Add your chunks from this morning
# Note: ChromaDB can handle the "Embedding" for you automatically using
# a default model (all-MiniLM-L6-v2) if you don't provide your own.
collection.add(
    documents=[
        "The Apple iPhone 16 features a titanium frame and a new A18 chip.",
        "The Granny Smith Apple is known for its tart flavor and green skin.",
        "The weather in Paris is beautiful in the spring.",
        "Deep learning is a subset of machine learning",
    ],
    metadatas=[
        {"type": "tech"},
        {"type": "fruit"},
        {"type": "travel"},
        {"type": "education"},
    ],
    ids=["id1", "id2", "id3", "id4"],
)

collection.update(ids=["id1"], metadatas=[{"type": "technology", "updated": True}])

print("Collection created and data 'Upserted'!")

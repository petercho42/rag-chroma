from database import collection

# 1. Query for something that doesn't use the exact words in the DB
results = collection.query(
    query_texts=["Tell me about some high-end hardware"],
    n_results=1,  # This is your 'Top K'
)

print(f"Result: {results['documents']}")
print(f"Distance Score: {results['distances']}")

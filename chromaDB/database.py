# database_manager.py
import chromadb
import os

# Use an absolute path to be 100% sure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "persistent_storage")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name="my_rag_docs")

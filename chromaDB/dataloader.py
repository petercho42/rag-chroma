import ollama
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. CONNECT TO YOUR 3090
SERVER_IP = "szuni"  # or your IP address
CHROMA_PORT = 8001

chroma_client = chromadb.HttpClient(host=SERVER_IP, port=CHROMA_PORT)
ollama_client = ollama.Client(host=f"http://{SERVER_IP}:11434")

# Create the collection on the 3090
collection = chroma_client.get_or_create_collection(name="personal_knowledge_base")


def process_and_upload(file_path):
    # STEP A: Mac reads the PDF
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    # STEP B: Mac splits it into chunks (The Sliding Window!)
    # chunk_overlap ensures we don't cut a sentence in half
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(pages)

    print(f"Uploading {len(chunks)} chunks to the 3090...")

    # STEP C: Mac coordinates the 3090's work
    for i, chunk in enumerate(chunks):
        embed_resp = ollama_client.embeddings(model="nomic-embed-text", prompt=chunk.page_content)
        
        # We extract the page number from LangChain's chunk metadata
        page_num = chunk.metadata.get("page", 0) 

        collection.add(
            ids=[f"{file_path}_{i}"],
            embeddings=[embed_resp["embedding"]],
            documents=[chunk.page_content],
            # We store the page number so we can filter by it later!
            metadatas=[{"page": page_num, "source": file_path}] 
        )
    print(f"Successfully indexed: {file_path}")


if __name__ == "__main__":
    # Test it with one file in your source_docs folder
    process_and_upload("./chromaDB/source_docs/NIPS-2017-attention-is-all-you-need-Paper.pdf")

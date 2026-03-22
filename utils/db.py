import chromadb
from chromadb.config import Settings

# Setup ChromaDB
CHROMA_PATH = r"C:\Users\sargu\OneDrive\Desktop\Final_year_project\ReCNA\dataset\chroma_db"
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)

SECTION_TYPES = ["introduction", "methodology", "discussion", "conclusion"]

global_collection = chroma_client.get_or_create_collection(name="paper_embeddings")
section_collection = chroma_client.get_or_create_collection(name="section_embeddings")
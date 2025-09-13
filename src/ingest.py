# src/ingest.py
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")

def load_documents():
    """Load all documents from the data/ directory"""
    loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
    docs = loader.load()
    return docs

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for vectorization"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_documents()
    chunks = split_documents(docs)
    print(f"Loaded {len(docs)} documents")
    print(f"Split into {len(chunks)} chunks")

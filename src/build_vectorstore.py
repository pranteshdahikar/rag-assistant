# src/build_vectorstore.py

import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")


def load_documents():
    """Load all .txt documents from data folder"""
    print("🔹 Loading documents from 'data/' folder...")
    all_docs = []
    for file_name in os.listdir(DATA_PATH):
        if file_name.endswith(".txt"):
            file_path = os.path.join(DATA_PATH, file_name)
            loader = TextLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
            print(f"  Loaded {len(docs)} document(s) from {file_name}")
    print(f"✅ Total documents loaded: {len(all_docs)}\n")
    return all_docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    """Split documents into chunks for embedding"""
    print("🔹 Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}\n")
    return chunks


def build_vectorstore():
    """Build FAISS vectorstore from document chunks"""
    docs = load_documents()
    if len(docs) == 0:
        print("⚠️ No documents found in data/. Add .txt files before building vectorstore.")
        return

    chunks = split_documents(docs)

    print("🔹 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("🔹 Building FAISS vectorstore...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print(f"🔹 Saving vectorstore to '{VECTORSTORE_PATH}'...")
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("✅ Vectorstore saved successfully!")


if __name__ == "__main__":
    build_vectorstore()

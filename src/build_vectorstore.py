# src/build_vectorstore.py

import os
import json
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document  # to create documents manually

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")


def load_documents():
    """Load all .txt or .json documents from data folder with metadata (filename)"""
    print("🔹 Loading documents from 'data/' folder...")
    all_docs = []

    for file_name in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file_name)

        if file_name.endswith(".txt") or file_name.endswith(".json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # If JSON, parse and extract 'publication_description'
                if file_name.endswith(".json") or content.strip().startswith("["):
                    data = json.loads(content)
                    for item in data:
                        text = item.get("publication_description", "")
                        if text:
                            doc = Document(page_content=text, metadata={"source": file_name})
                            all_docs.append(doc)
                else:
                    # Plain text fallback
                    loader = TextLoader(file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = file_name
                    all_docs.extend(docs)

                print(f"  Loaded {len(all_docs)} document(s) from {file_name}")

            except Exception as e:
                print(f"⚠️ Failed to load {file_name}: {e}")

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
        print("⚠️ No documents found in data/. Add .txt or JSON files before building vectorstore.")
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

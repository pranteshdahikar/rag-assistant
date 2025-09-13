# src/rag_assistant.py

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

# Paths
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

def load_rag_chain():
    """
    Load FAISS vectorstore and initialize a local Hugging Face LLM
    with chunked retrieval for longer documents.
    """
    # Initialize embeddings (must match build_vectorstore.py)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load FAISS vectorstore with explicit permission for pickle deserialization
    print("🔹 Loading FAISS vectorstore...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True  # safe for locally created vectorstore
    )

    # Initialize local Hugging Face LLM
    print("🔹 Initializing local Hugging Face LLM...")
    hf_pipe = pipeline(
        "text-generation",
        model="google/flan-t5-small",  # small, fast, free model
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    # RetrievalQA chain with chunked retrieval
    print("🔹 Setting up RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})  # retrieve top 3 relevant chunks
    )

    return qa_chain

def main():
    qa_chain = load_rag_chain()
    print("✅ RAG assistant is ready! Ask questions about your documents.")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Query the RAG chain
        result = qa_chain.run(query)
        print("\n📄 Answer:\n", result)

if __name__ == "__main__":
    main()

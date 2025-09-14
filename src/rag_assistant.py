# src/rag_assistant.py

import os
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from transformers import pipeline

# Paths
VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# In-memory session memory
SESSION_MEMORY = {}


def load_rag_chain(filter_source: str = None):
    """
    Load FAISS vectorstore and initialize a local Hugging Face LLM
    with optional filtering by document source (e.g., 'doc1.txt').
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("🔹 Loading FAISS vectorstore...")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    print("🔹 Initializing local Hugging Face LLM...")
    hf_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=256
    )
    llm = HuggingFacePipeline(pipeline=hf_pipe)

    retriever_kwargs = {"k": 3}
    if filter_source:
        retriever_kwargs["filter"] = {"source": filter_source}
        print(f"🔹 Restricting retrieval to: {filter_source}")

    retriever = vectorstore.as_retriever(search_kwargs=retriever_kwargs)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain


def modify_question_with_memory(user_id: str, new_question: str) -> str:
    """
    Modify the user's new question by using session memory to make it standalone.
    """
    past_questions: List[str] = SESSION_MEMORY.get(user_id, [])
    if past_questions:
        context = " ".join(past_questions)
        # Simple approach: prepend past questions to the new question
        standalone_question = f"Based on previous questions: {context} New question: {new_question}"
    else:
        standalone_question = new_question
    return standalone_question


def main():
    print("✅ RAG assistant is ready! Ask questions about your documents.")
    print("💡 Tip: Use 'doc:filename.txt your question' to restrict search to a file.")
    print("💡 Tip: Use 'memory:<user_id> your question' to track memory per user.\n")

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # Parse memory mode
        user_id = None
        if query.startswith("memory:"):
            try:
                parts = query.split(" ", 1)
                user_id = parts[0].replace("memory:", "").strip()
                query = parts[1].strip()
            except IndexError:
                print("⚠️ Usage: memory:<user_id> your question")
                continue

        # Parse document filter
        filter_source = None
        if query.startswith("doc:"):
            try:
                parts = query.split(" ", 1)
                filter_source = parts[0].replace("doc:", "").strip()
                query = parts[1].strip()
            except IndexError:
                print("⚠️ Usage: doc:filename.txt your question")
                continue

        # If memory is enabled, modify question
        if user_id:
            query = modify_question_with_memory(user_id, query)

        qa_chain = load_rag_chain(filter_source=filter_source)
        result = qa_chain.invoke({"query": query})
        answer = result["result"]

        # Update session memory
        if user_id:
            SESSION_MEMORY.setdefault(user_id, []).append(query)

        print("\n📄 Answer:\n", answer)


if __name__ == "__main__":
    main()

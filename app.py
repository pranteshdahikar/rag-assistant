import os
from src.build_vectorstore import build_vectorstore
from src.rag_assistant import load_rag_chain

def main():
    # Step 1: Build vectorstore if it doesn't exist
    VECTORSTORE_PATH = os.path.join(os.path.dirname(__file__), "vectorstore")
    if not os.path.exists(VECTORSTORE_PATH):
        print("Vectorstore not found, building...")
        build_vectorstore()
    else:
        print("Vectorstore found. Skipping build.")

    # Step 2: Load RAG assistant
    qa_chain = load_rag_chain()
    print("✅ RAG assistant is ready! Ask questions about your documents.")

    # Step 3: Interactive CLI
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        result = qa_chain.run(query)
        print("\n📄 Answer:\n", result)

if __name__ == "__main__":
    main()

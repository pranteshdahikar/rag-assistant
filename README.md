# RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) assistant built with **LangChain**, **FAISS**, and **Hugging Face** models.  
It lets you query your local documents (`.txt` and `.pdf`) in natural language.

---

## 🚀 Installation

## Setup Instructions

1. **Clone the repository**

git clone https://github.com/your-username/rag-assistant.git
cd rag-assistant


2. Create a virtual environment (optional but recommended)

python -m venv venv
# Windows
venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

4. Prepare your documents
Place .txt or .json files inside the data/ folder.
JSON files should contain text fields such as title or publication_description.

5. Build the vectorstore
python src/build_vectorstore.py

6. Run the RAG assistant
python src/rag_assistant.py

### Usage Examples

✅ RAG assistant is ready! Ask questions about your documents.
💡 Tip: Use 'doc:filename.txt your question' to restrict search to a file.
💡 Tip: Use 'memory:<user_id> your question' to track memory per user.


Ask a question (or type 'exit'): What tools are mentioned?
🔹 Loading FAISS vectorstore...
🔹 Initializing local Hugging Face LLM...
Device set to use cpu

📄 Answer:
 Toolboxes

Ask a question (or type 'exit'): What’s this publication about?
🔹 Loading FAISS vectorstore...
🔹 Initializing local Hugging Face LLM...
Device set to use cpu

📄 Answer:
 Science/Tech

Ask a question (or type 'exit'): Any limitations or assumptions?
🔹 Loading FAISS vectorstore...
🔹 Initializing local Hugging Face LLM...
Device set to use cpu

📄 Answer:
 A section that discusses the limitations of the dataset and the Mini-RAG system, addressing any potential issues, trade-offs, and the impact of any limitations on the research outcomes.


### How It Works
1.	Documents are loaded and split into chunks.
2.	Each chunk is embedded using Hugging Face embeddings.
3.	A FAISS vectorstore is created for fast similarity search.
4.	User queries are sent through a RetrievalQA chain:
•	Retrieves top-k relevant chunks
•	Uses a local Hugging Face LLM to generate answers
•	Optional: Filters retrieval by document source
•	Optional: Uses past questions to generate standalone queries

## Notes
•	Use exit or quit to stop the assistant.
•	JSON documents must include meaningful text fields for embeddings.
•	You can change the model in src/rag_assistant.py if desired.
•	The system prompt for modifying questions can be customized.
•	For memory-enabled RAG, use MongoDB or other database integration.



## Dependencies:

•	Python 3.10+
•	langchain
•	langchain-huggingface
•	transformers
•	faiss-cpu
•	python-dotenv (optional, if using OpenAI API)
•	pymongo (optional, for memory storage)

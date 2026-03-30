# AI Knowledge Assistant (Local RAG Pipeline)

A fully localized, open-source Retrieval-Augmented Generation (RAG) Document Ingestion pipeline.

This project is built using the modern **LangChain** ecosystem and completely bypasses the need for an OpenAI API key. By leveraging HuggingFace's lightweight `all-MiniLM-L6-v2` embedding model, all data processing, vectorization, and searching is done entirely on your local machine using your own CPU/GPU hardware.

## Project Structure

```
ai-knowledge-assistant/
├── app/
│   ├── chroma_db/       # Persistent local vector database (created automatically)
│   ├── data/            # Place your PDF documents here!
│   └── src/             
│       ├── ingest.py    # Reads PDFs, chunks them, and embeds them into Chroma
│       ├── query.py     # Simple CLI testing script to query your database
│       └── test_reterival.py # Advanced retrieval utilizing LangChain's native Retriever architecture
├── .gitignore           # Ignores venv and database files
├── requirements.txt     # Python package dependencies
```

## Setup Instructions

**1. Create a Virtual Environment**
It's highly recommended to use a virtual environment so these heavy machine-learning packages don't conflict with your global python setup:
```bash
python3 -m venv venv
source venv/bin/activate
```

**2. Install Dependencies**
Install the necessary requirements (PyTorch, Transformers, LangChain, ChromaDB):
```bash
pip install -r requirements.txt
```

## Usage Guide

### Step 1: Add your Documents
Place any local `.pdf` files you want your AI assistant to read into the `app/data/` folder. 

### Step 2: Ingest the Documents
Run the ingestion script to process the documents. This script reads the PDFs, splits them into functional chunks (800 characters with 80 character overlap), generates vector embeddings for each chunk, and saves them sequentially into `app/chroma_db/`.
```bash
python app/src/ingest.py
```
*(Note: On the first run, the open-source `all-MiniLM-L6-v2` model weights (~90MB) will be securely downloaded to your computer under the hood and permanently cached).*

### Step 3: Query the Knowledge Base
You can now test out the RAG pipeline by querying the local database directly. We provide two test scripts:

**A. Simple Querying (`query.py`)**  
Allows you to pass a question directly via terminal arguments and get top-matching text excerpts:
```bash
python app/src/query.py "What is this document about?"
```

**B. LangChain Native Retriever (`test_reterival.py`)**  
A script demonstrating how to wrap your local Chroma DB into LangChain's standardized `retriever` pattern, making it ready to hook up to any Local LLMs (e.g. Ollama, Llama.cpp) down the road!
```bash
python app/src/test_reterival.py
```

## Future Expansions
- Next up, you could hook up [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) to this project to expose the functionality as a REST Web Server.
- You can pair the retriever with a local LLM via tools like `Ollama` to feed these retrieved chunks as context and generate conversational answers without the internet!

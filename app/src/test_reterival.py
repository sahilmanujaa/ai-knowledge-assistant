import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

# Load existing vector database with local HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# Create retriever (this is standard LangChain!)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test query
query = "What is this document about?"

docs = retriever.invoke(query)

print("\nTop Retrieved LangChain Chunks:\n")

for i, doc in enumerate(docs):
    print(f"Result {i+1}")
    print(doc.page_content.strip())
    print("-" * 50)

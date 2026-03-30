import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

def query_database(query_text):
    print(f"Searching for: '{query_text}'\n")
    
    # Ensure we use the exact same embedding model used during ingestion
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the persistent Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Perform a similarity search
    results = db.similarity_search_with_score(query_text, k=3)
    
    if not results:
        print("No matches found in the database. Are you sure you ran the ingestion script?")
        return
        
    print("--- Top Matches ---\n")
    for i, (doc, score) in enumerate(results):
        # A lower score generally means a closer match in LangChain's default Chroma distance metric (L2 usually)
        print(f"[{i+1}] Source: {doc.metadata.get('source', 'Unknown')} (Distance Score: {score:.4f})")
        print(f"Text Preview: {doc.page_content.strip()}")
        print("-" * 50)

if __name__ == "__main__":
    # Setup argparse so you can pass custom queries from the terminal
    parser = argparse.ArgumentParser(description="Query the local RAG database.")
    parser.add_argument(
        "query", 
        type=str, 
        nargs="?", 
        default="What is the purpose of this document?", 
        help="The question or text you want to search for."
    )
    args = parser.parse_args()
    
    query_database(args.query)

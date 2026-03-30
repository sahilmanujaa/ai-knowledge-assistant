import os
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def query_rag(query_text):
    print(f"Searching for: '{query_text}'\n")
    
    # Ensure we use the exact same embedding model used during ingestion
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the persistent Chroma database
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # Perform a similarity search
    results = db.max_marginal_relevance_search(
        query_text,
        k=5,
        fetch_k=20
    )
    
    if not results:
        print("No matches found in the database. Are you sure you ran the ingestion script?")
        return
        
    print("--- Top Matches ---\n")
    for i, doc in enumerate(results):

        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        print(f"\nResult {i+1}")
        print(f"Source File: {os.path.basename(source)}")
        print(f"Page: {page}")
        print(doc.page_content)
        print("="*80)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc in results]
    )

    prompt_template = ChatPromptTemplate.from_template(
        PROMPT_TEMPLATE
    )

    prompt = prompt_template.format(
        context=context_text,
        question=query_text
    )

    model = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY
    )

    response = model.invoke(prompt)

    print("\n🤖 AI Answer:\n")
    print(response.content)

    print("\n📚 Sources:\n")

    for doc in results:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")

        print(f"{os.path.basename(source)} | Page {page}")

if __name__ == "__main__":
    # Setup argparse so you can pass custom queries from the terminal
    query_text = input("Ask a question: ")
    query_rag(query_text)

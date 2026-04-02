from langchain_openai import ChatOpenAI
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

# Embeddings (matching what was used during ingestion)
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Vector store
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)

# Metadata schema
metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="Topic of the document. Valid values: python, mysql, docker, general",
        type="string"
    ),
    AttributeInfo(
        name="file",
        description="Original name of the PDF file",
        type="string"
    ),
]

document_content_description = "Technical documents about python, mysql, and docker"

# Self Query Retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
)

def main():
    print("Self-Query Retriever initialized.")
    print("Try asking things like: 'Show me documents about python' or 'Give me information about docker'")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not query:
            continue

        print(f"\nQuerying for: '{query}'...")
        docs = retriever.invoke(query)

        if not docs:
            print("No documents found.")
        else:
            print(f"Found {len(docs)} documents:")
            for i, d in enumerate(docs):
                print(f"--- Document {i+1} ---")
                print(f"Content: {d.page_content.strip()[:150]}...")
                print(f"Metadata: {d.metadata}")
                
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()

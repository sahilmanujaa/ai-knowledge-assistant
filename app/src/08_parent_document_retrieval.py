"""
8 — Parent Document Retrieval Pipeline
=============================================

Technique: Parent Document Retrieval (Small to Big Retrieval)

High-Level Flow:
─────────────────────────────────────────────────────
  Raw Document
        ↓
  Large Text Splitter (e.g., 2000 chars) -> Parent Chunks
  Small Text Splitter (e.g., 400 chars)  -> Child Chunks
        ↓
  Vector Store (Chroma)       <-- Stores Embeddings of Child Chunks
  Document Store (InMemory)   <-- Stores Parent Chunks
        ↓
  User Question
        ↓
  Similarity Search on Vector Store
  [Retrieves the most relevant small Child Chunks for precision]
        ↓
  Parent Document Retriever
  [Looks up the Parent Chunk corresponding to the retrieved Child Chunk]
  [Passes the entire Parent Chunk to the LLM to provide full context]
        ↓
  ChatPromptTemplate + LLM (gpt-4o-mini)
        ↓
  Grounded, Context-Rich Answer
─────────────────────────────────────────────────────

Use Cases:
----------
1. Complex Technical Documents: When the answer to a query requires understanding the surrounding context (e.g., code blocks and their preceding explanations), retrieving small chunks might miss the narrative, but querying large chunks might dilute similarity search accuracy.
2. Legal Contracts: Small clauses are easy to match against a query, but the LLM needs the entire section to accurately interpret the clause's implications. 
3. Medical Guidelines: Finding the specific symptom in a small chunk is fast, but returning the whole disease protocol chunk prevents hallucinating treatments.
4. "Lost in the Middle" Prevention: Minimizing the noise in similarity search while maximizing the context window utility for the final prompt generation.

Run:
    python src/08_parent_document_retrieval.py
"""

import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base_dir, "data")
# We use a separate vectorstore directory so it doesn't conflict with our standard ingestion
CHROMA_PATH = os.path.join(base_dir, "chroma_db_parent")

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def init_retriever():
    print("Setting up Parent Document Retriever...\n")
    
    # 1. Load Documents
    print(f"Loading PDFs from {DATA_PATH}...")
    loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    if not docs:
        print("No documents found in the data directory!")
        return None
    
    # 2. Splitters
    # Parent (large chunks)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    # Child (smaller chunks for accurate retrieval)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    # 3. Embedding Function
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Vector Store (for small child chunks)
    vectorstore = Chroma(
        collection_name="split_parents", 
        persist_directory=CHROMA_PATH,
        embedding_function=embedding
    )

    # 5. Document Store (for storing the large parent chunks)
    store = InMemoryStore()

    # 6. Initialize Retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        # Fetch top 4 child documents, then figure out which parents they belong to
        search_kwargs={"k": 4} 
    )

    print("Ingesting documents (This may take a moment)...")
    # This will split into parents, then children, build embeddings for children,
    # save children in Chroma, and save parents in the InMemoryStore.
    retriever.add_documents(docs)
    print("Ingestion complete.\n")
    
    return retriever

def main():
    retriever = init_retriever()
    if not retriever:
        return
        
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    print("Parent Document Retriever is ready.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not query:
            continue

        print(f"\nRetrieving context for: '{query}'...")
        
        # This performs the similarity search for Child chunks, 
        # retrieves their IDs, maps them to Parent chunks, 
        # and returns the Parent chunks!
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            print("No relevant context found.")
            continue
            
        print(f"Found {len(retrieved_docs)} parent chunks.")
        
        # Build context
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in retrieved_docs]
        )
        
        # Generate Answer
        prompt = prompt_template.format(context=context_text, question=query)
        response = llm.invoke(prompt)

        print("\n🤖 AI Answer:\n")
        print(response.content)

        print("\n📚 Sources (Parent Documents):\n")
        for i, doc in enumerate(retrieved_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            print(f"{i+1}. {os.path.basename(source)} | Page {page} (Length: {len(doc.page_content)} characters)")
            
        print("-" * 60 + "\n")

if __name__ == "__main__":
    main()

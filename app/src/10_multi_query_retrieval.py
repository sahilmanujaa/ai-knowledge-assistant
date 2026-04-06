"""
10 — Multi-Query Retrieval Pipeline
======================================================

Technique: Multi-Query Retrieval (LangChain MultiQueryRetriever)

High-Level Flow:
─────────────────────────────────────────────────────
  User Question
        ↓
  Query Generator (LLM)
  [MultiQueryRetriever uses the LLM to rephrase the original
   question into multiple semantically diverse variants]
        ↓
  Multiple Queries  (generated internally by MultiQueryRetriever)
        ↓
  Vector Search (each query independently against ChromaDB)
  [Each query variant hits the vector store separately]
        ↓
  Merge Results (automatic deduplication by MultiQueryRetriever)
  [All retrieved documents are pooled & duplicates removed]
        ↓
  Final Context  (richer, more diverse set of relevant chunks)
        ↓
  LLM Answer
  [LLM synthesises a grounded answer from the merged context]
─────────────────────────────────────────────────────

Why Multi-Query?
----------------
A single embedding query may miss relevant chunks because the user's
phrasing doesn't align with how the information was written.
MultiQueryRetriever automatically generates multiple reformulations
of the same question, which:

1. Overcomes Vocabulary Mismatch: Different phrasings catch chunks
   that a single query would miss due to embedding space sensitivity.
2. Increases Recall: More diverse queries surface more relevant
   documents, reducing the risk of an incomplete answer.
3. Reduces Fragility: Robust to ambiguous or poorly-worded questions
   because the LLM clarifies intent before searching.

Run:
    python src/10_multi_query_retrieval.py
"""

import os
import logging
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & Paths
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

# Suppress noisy ChromaDB / tokeniser logs
logging.getLogger("chromadb").setLevel(logging.ERROR)

# Enable LangChain's internal logging so we can see the generated queries
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Prompt Template
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """Answer the question based only on the following context.

{context}

---

Answer the question based on the above context: {question}
"""


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------
def init_components():
    """Set up LLM, vector store and MultiQueryRetriever."""
    print("Initialising Multi-Query Retrieval pipeline...\n")

    # LLM — used both for query generation and final answer
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # Embeddings & Vector Store
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
    )

    # Base retriever (standard similarity search)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Multi-Query Retriever
    # Internally: LLM generates multiple query variants → runs each against
    # base_retriever → merges & deduplicates results automatically.
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm,
    )

    print("✅ MultiQueryRetriever is ready.\n")
    return multi_query_retriever, llm


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------
def main():
    try:
        retriever, answer_llm = init_components()
    except Exception as e:
        print(f"❌ Failed to initialise pipeline: {e}")
        print("Note: Make sure you have populated the Chroma DB using 01_ingest.py!")
        return

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    print("Type 'quit', 'exit', or 'q' to stop.\n")
    print("=" * 70)

    while True:
        query = input("\nEnter your query: ").strip()

        if query.lower() in {"quit", "exit", "q"}:
            print("Exiting...")
            break

        if not query:
            continue

        # ── Step 1 & 2: MultiQueryRetriever handles query generation,
        #                vector search, and merging automatically ──────────
        print(f"\n[1] Running MultiQueryRetriever for: '{query}'")
        print("(LLM will generate query variants — see INFO logs above)\n")

        try:
            docs = retriever.invoke(query)
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            continue

        if not docs:
            print("❌ No relevant documents found. Try a different query.")
            continue

        print(f"[2] Retrieved {len(docs)} unique document chunks (after deduplication)\n")

        # ── Step 3: Build context from merged docs ─────────────────────────
        context_text = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # ── Step 4: Generate final answer ─────────────────────────────────
        print("[3] Generating final answer...")
        try:
            prompt = prompt_template.format(context=context_text, question=query)
            response = answer_llm.invoke(prompt)
        except Exception as e:
            print(f"❌ Answer generation failed: {e}")
            continue

        # ── Output ────────────────────────────────────────────────────────
        print("\n🤖 AI Answer:\n")
        print(response.content)

        print("\n📚 Retrieved Sources:\n")
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            page   = doc.metadata.get("page", "Unknown")
            print(f"--- Chunk {i + 1} ({os.path.basename(source)} | Page {page}) ---")
            # Print a preview (first 300 chars) to keep output manageable
            preview = doc.page_content.strip()
            print(preview[:300] + ("..." if len(preview) > 300 else ""))
            print()

        print("=" * 70)


if __name__ == "__main__":
    main()

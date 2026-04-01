"""
4 (Extended of 3) — Query Routing with Per-Topic Retrievers
=============================================================

Technique: LLM Router → Topic-Filtered Chroma Retriever → Conversational QA

High-Level Flow:
─────────────────────────────────────────────────────
  User Question
        ↓
  Query Router  (LLM — gpt-4o-mini)
  [Classifies question → python | mysql | docker]
  [Falls back to 'python' on unknown output]
        ↓
  Topic-Filtered Retriever  (one per topic, pre-built)
  [Chroma filter: {"topic": "<routed_source>"}]
  [Searches only the relevant partition of the shared DB]
  [k=4 chunks returned]
        ↓
  ConversationalRetrievalChain  (per-topic, pre-built)
  [Condenses chat_history + question → standalone query]
  [Retrieves topic-filtered chunks]
  [Shared ConversationBufferMemory across all topics]
        ↓
  ChatOpenAI  (gpt-4o-mini)
  [Generates answer using only relevant topic's context]
        ↓
  Answer  (with routing label shown e.g. [Router] → docker)
─────────────────────────────────────────────────────

Key difference from query_memory.py:
  Questions are routed to the correct knowledge domain first,
  avoiding cross-topic noise in retrieval.

Run:
    python src/query_router.py
"""

import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")   # Single shared DB
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Valid topic names — must match the 'topic' metadata set in ingest.py
VALID_SOURCES = ["python", "mysql", "docker"]
DEFAULT_SOURCE = "python"

# ─────────────────────────────────────────────
# Step 1 — Router
# ─────────────────────────────────────────────

ROUTER_PROMPT = """
You are a query router.

Your job is to decide which knowledge base should answer the question.

Possible sources:
- python
- mysql
- docker

Return ONLY the source name, nothing else.

Question: {question}
"""

def route_query(question: str, llm: ChatOpenAI) -> str:
    """
    Routes the question to the appropriate topic name.
    Falls back to DEFAULT_SOURCE if the LLM returns an unexpected value.
    """
    response = llm.invoke(
        [HumanMessage(content=ROUTER_PROMPT.format(question=question))]
    )
    source = response.content.strip().lower()

    if source not in VALID_SOURCES:
        print(f"[Warning] Router returned unknown source '{source}'. Falling back to '{DEFAULT_SOURCE}'.")
        return DEFAULT_SOURCE

    return source


# ─────────────────────────────────────────────
# Step 2 — Load Single Chroma DB
# ─────────────────────────────────────────────
# A single chroma_db holds all chunks.
# Each chunk has a 'topic' metadata field set during ingestion.
# We filter by topic at retrieval time using Chroma's `where` filter.

def create_retriever_map(db: Chroma) -> dict:
    """
    Creates one retriever per topic — each filters the shared DB
    to only return chunks whose metadata topic matches.
    """
    retriever_map = {}
    for source in VALID_SOURCES:
        retriever_map[source] = db.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"topic": source}   # Chroma metadata filter
            }
        )
        print(f"  ✓ Retriever ready for topic: {source}")
    return retriever_map


# ─────────────────────────────────────────────
# Step 3 — Main Chat Loop with Router
# ─────────────────────────────────────────────

def main():
    print("Starting Routed AI Knowledge Assistant...\n")

    print("Loading embeddings model...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print(f"Connecting to Chroma DB at {CHROMA_PATH}...")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    retriever_map = create_retriever_map(db)

    print("Initializing LLM and Memory...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    # Pre-build one QA chain per topic — avoids recreating on every query
    chain_map = {
        source: ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        for source, retriever in retriever_map.items()
    }

    print("\nReady! Ask anything.\n")

    while True:
        query = input("Ask a question (or type 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Route the query and pick the pre-built chain
        source = route_query(query, llm)
        print(f"[Router] → {source}")

        result = chain_map[source].invoke({"question": query})

        print("\nAnswer:")
        print(result["answer"])

        # Uncomment to show source documents:
        # print("\nSources:")
        # for doc in result["source_documents"]:
        #     print(f"  {doc.metadata.get('file', 'unknown')} | topic: {doc.metadata.get('topic', '-')}")

        print()


if __name__ == "__main__":
    main()

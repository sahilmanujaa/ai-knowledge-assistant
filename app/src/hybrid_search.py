"""
4 (Advanced) — Hybrid Search  (BM25 + Vector Ensemble)
=============================================================

Technique: Keyword Search (BM25) + Semantic Search (MMR) → Ensemble Fusion

High-Level Flow:
─────────────────────────────────────────────────────
  User Question
        ↓
  ┌─────────────────────┐   ┌──────────────────────────┐
  │   BM25Retriever     │   │  Chroma Vector Retriever  │
  │  (keyword / TF-IDF) │   │  (MMR, k=3, fetch_k=10)  │
  │  built from all     │   │  semantic similarity on   │
  │  Chroma documents   │   │  all-MiniLM-L6-v2 embeds  │
  └─────────┬───────────┘   └────────────┬─────────────┘
            │  k=3 chunks                │  k=3 chunks
            └──────────────┬─────────────┘
                           ↓
              EnsembleRetriever
              [Reciprocal Rank Fusion — weights 0.5 / 0.5]
              [Merges & deduplicates results from both]
                           ↓
              ConversationalRetrievalChain
              [Condenses history + question → standalone query]
              [Memory: ConversationBufferMemory]
                           ↓
              ChatOpenAI  (gpt-4o-mini)
              [Answers using hybrid-retrieved context]
                           ↓
              Answer + Source metadata
─────────────────────────────────────────────────────

Why hybrid?
  BM25  → exact keyword match  (e.g. function names, error codes)
  Vector → conceptual similarity  (e.g. synonyms, paraphrasing)
  Together they outperform either method alone.

Run:
    python src/hybrid_search.py
"""

import os
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

from langchain_chroma import Chroma

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------
# Initialize Embeddings & DB
# -----------------------------
print("Initializing embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db_path = os.path.join(base_dir, "chroma_db")
print("Connecting to Chroma database...")
vector_db = Chroma(
    persist_directory=vector_db_path,
    embedding_function=embeddings
)

# Extract documents from Chroma for BM25
data = vector_db.get()
chunks = []
if data and data['documents']:
    for text, meta in zip(data['documents'], data['metadatas']):
        chunks.append(Document(page_content=text, metadata=meta))
print(f"Loaded {len(chunks)} document chunks for BM25 index.")


# -----------------------------
# Vector Retriever
# -----------------------------
vector_retriever = vector_db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10}
)


# -----------------------------
# BM25 Retriever
# -----------------------------
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3


# -----------------------------
# Hybrid Retriever
# -----------------------------
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)


# -----------------------------
# LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY,
    temperature=0
)


# -----------------------------
# Memory
# -----------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)


# -----------------------------
# RAG Chain
# -----------------------------
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)


# -----------------------------
# Chat Loop
# -----------------------------
print("\nAI Knowledge Assistant Ready!\n")

while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    result = qa_chain.invoke({"question": query})

    print("\nAnswer:\n")
    print(result["answer"])

    print("\nSources:\n")

    for doc in result["source_documents"]:
        print(doc.metadata)
        print("-" * 80)

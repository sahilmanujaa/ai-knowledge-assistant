"""
5 — Reranking Retrieval (Improving Accuracy After Retrieval)
=================================================================

Pipeline:
    User Question
          ↓
    Query Router          (LLM picks: python | mysql | docker)
          ↓
    Vector DB Retrieval   (k=20 broad fetch from Chroma)
          ↓
    Reranking Model       (cross-encoder: BAAI/bge-reranker-base).
          ↓
    Top 4 Chunks          (highest relevance score only)
          ↓
    LLM                   (GPT-4o-mini generates final answer)
          ↓
    Answer

Why reranking?
  - Vector similarity ≠ semantic relevance ranking
  - Cross-encoder reads query + chunk *together* — much richer signal
  - Accuracy improvement: 20–50% over plain retrieval
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from sentence_transformers import CrossEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Valid topic names matching the 'topic' metadata set in ingest.py
VALID_SOURCES = ["python", "mysql", "docker"]
DEFAULT_SOURCE = "python"

# ── Retrieval settings ────────────────────────────────────────────────────────
INITIAL_K = 20      # broad fetch — cast a wide net first
FINAL_K   = 4       # top N after reranking — only the best reach the LLM

# ── Reranker model ────────────────────────────────────────────────────────────
# BAAI/bge-reranker-base  — open-source cross-encoder, runs locally
# Alternatives: cross-encoder/ms-marco-MiniLM-L-6-v2, cohere rerank (API)
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Query Router
# ─────────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """\
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
    Uses the LLM to classify which topic (python / mysql / docker)
    the question belongs to.  Falls back to DEFAULT_SOURCE on unknown output.
    """
    response = llm.invoke(
        [HumanMessage(content=ROUTER_PROMPT.format(question=question))]
    )
    source = response.content.strip().lower()

    if source not in VALID_SOURCES:
        print(
            f"[Router] Unknown source '{source}' — "
            f"falling back to '{DEFAULT_SOURCE}'"
        )
        return DEFAULT_SOURCE

    return source


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Broad Vector Retrieval  (k=20)
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_candidates(
    question: str,
    db: Chroma,
    source: str,
    k: int = INITIAL_K,
) -> List[Document]:
    """
    Retrieves k=20 candidate chunks from Chroma, pre-filtered to the
    topic selected by the router.  Wide net → reranker applies precision.
    """
    docs = db.similarity_search(
        query=question,
        k=k,
        filter={"topic": source},   # metadata filter from ingest.py
    )
    print(f"[Retrieval] Fetched {len(docs)} candidate chunks (topic={source})")
    return docs


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Reranker  (cross-encoder scores every query + chunk pair)
# ─────────────────────────────────────────────────────────────────────────────

def rerank_documents(
    question: str,
    docs: List[Document],
    reranker: CrossEncoder,
    top_n: int = FINAL_K,
) -> List[Document]:
    """
    Scores each (query, chunk) pair with the cross-encoder, then returns
    only the top_n chunks sorted by descending relevance score.

    The cross-encoder reads both the query AND the document together,
    which captures much richer relevance signals than vector similarity alone.
    """
    # Build (query, chunk_text) pairs for batch scoring
    pairs: List[Tuple[str, str]] = [
        (question, doc.page_content) for doc in docs
    ]

    # Score all pairs in one forward pass (efficient batch inference)
    scores: List[float] = reranker.predict(pairs).tolist()

    # Zip scores → docs, sort descending, take top_n
    scored_docs = sorted(
        zip(scores, docs),
        key=lambda x: x[0],
        reverse=True,
    )

    top_docs = [doc for _score, doc in scored_docs[:top_n]]

    print(f"\n[Reranker] Scores for top {top_n} chunks:")
    for i, (score, doc) in enumerate(scored_docs[:top_n], start=1):
        source_file = doc.metadata.get("source", "unknown")
        preview = doc.page_content[:80].replace("\n", " ")
        print(f"  #{i}  score={score:.4f}  file={source_file}")
        print(f"       preview: {preview}...")

    return top_docs


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — LLM Answer Generation
# ─────────────────────────────────────────────────────────────────────────────

ANSWER_PROMPT = """\
You are a helpful AI knowledge assistant.

Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""


def generate_answer(
    question: str,
    top_docs: List[Document],
    llm: ChatOpenAI,
) -> str:
    """
    Concatenates the top reranked chunks into a single context string
    and asks the LLM to produce a grounded answer.
    """
    context = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{doc.page_content}"
        for i, doc in enumerate(top_docs)
    )

    prompt = ANSWER_PROMPT.format(context=context, question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Full Reranking RAG Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def reranked_rag(
    question: str,
    db: Chroma,
    reranker: CrossEncoder,
    llm: ChatOpenAI,
) -> dict:
    """
    End-to-end pipeline:
      1. Route query  → pick topic
      2. Retrieve k=20 candidates from Chroma
      3. Rerank with cross-encoder → keep top 4
      4. Generate answer via LLM

    Returns a dict with keys: source, candidates, top_docs, answer
    """
    print("\n" + "=" * 60)
    print(f"Question: {question}")
    print("=" * 60)

    # 1 — Route
    source = route_query(question, llm)
    print(f"[Router]  → {source}")

    # 2 — Broad retrieval
    candidates = retrieve_candidates(question, db, source, k=INITIAL_K)

    if not candidates:
        return {
            "source": source,
            "candidates": [],
            "top_docs": [],
            "answer": "No relevant documents found in the knowledge base.",
        }

    # 3 — Rerank
    top_docs = rerank_documents(question, candidates, reranker, top_n=FINAL_K)

    # 4 — LLM answer
    answer = generate_answer(question, top_docs, llm)

    return {
        "source": source,
        "candidates": candidates,
        "top_docs": top_docs,
        "answer": answer,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main — Interactive Chat Loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Day 21 — Reranking RAG Pipeline")
    print("  Vector DB (k=20) → Cross-Encoder → Top 4 → LLM")
    print("=" * 60)

    # ── Load embedding model ─────────────────────────────────────────────────
    print("\n[Init] Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # ── Connect to Chroma ────────────────────────────────────────────────────
    print(f"[Init] Connecting to Chroma DB at: {CHROMA_PATH}")
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

    # ── Load reranker model (runs locally, no API key needed) ────────────────
    print(f"[Init] Loading reranker model: {RERANKER_MODEL}")
    print("       (first run downloads ~1 GB — subsequent runs use cache)")
    reranker = CrossEncoder(RERANKER_MODEL)

    # ── Initialize LLM ───────────────────────────────────────────────────────
    print("[Init] Initializing LLM (gpt-4o-mini)...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    print("\n✅ All components ready. Ask anything (type 'exit' to quit).\n")

    # ── Chat loop ─────────────────────────────────────────────────────────────
    while True:
        question = input("Ask a question: ").strip()

        if not question:
            continue

        if question.lower() == "exit":
            print("Goodbye!")
            break

        result = reranked_rag(question, db, reranker, llm)

        print("\n📌 Answer:")
        print(result["answer"])

        # Optional: show which sources were used
        print("\n📄 Sources used (top chunks after reranking):")
        for i, doc in enumerate(result["top_docs"], start=1):
            src = doc.metadata.get("source", "unknown")
            topic = doc.metadata.get("topic", "-")
            print(f"  [{i}] {src}  (topic: {topic})")

        print()


if __name__ == "__main__":
    main()

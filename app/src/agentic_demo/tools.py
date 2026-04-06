"""
tools.py — Tool Definitions for the ReAct Agent
================================================

Each function decorated with @tool becomes a callable action that the
LLM may choose to invoke during its reasoning loop.

Available Tools
---------------
1. knowledge_base_search  — Searches the existing ChromaDB vector store
2. calculator             — Safely evaluates a maths expression
3. get_current_datetime   — Returns the current date and time
4. word_count             — Returns word / char / sentence statistics

Design Notes
------------
- Every tool docstring IS the tool description the LLM reads when
  deciding which tool to use — keep them precise and action-oriented.
- Tools must accept and return plain strings so the ReAct loop can
  embed the observation directly into the prompt.
"""

import os
import re
import datetime
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------------------------
# ChromaDB — shared vector store from 01_ingest.py
# ---------------------------------------------------------------------------
_BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CHROMA_PATH = os.path.join(_BASE_DIR, "chroma_db")

_embedding   = None
_vectorstore = None


def _get_vectorstore() -> Chroma:
    """Lazy-initialise the vector store once, reuse on subsequent calls."""
    global _embedding, _vectorstore
    if _vectorstore is None:
        _embedding   = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectorstore = Chroma(
            persist_directory=_CHROMA_PATH,
            embedding_function=_embedding,
        )
    return _vectorstore


# ---------------------------------------------------------------------------
# Tool 1 — Knowledge Base Search
# ---------------------------------------------------------------------------
@tool
def knowledge_base_search(query: str) -> str:
    """
    Search the local knowledge base (ChromaDB vector store) for documents
    relevant to the query. Use this tool when you need factual information
    that may be stored in the ingested PDF documents. Input should be a
    plain-text search query string.
    """
    try:
        vs   = _get_vectorstore()
        docs = vs.similarity_search(query, k=3)
        if not docs:
            return "No relevant documents found in the knowledge base."
        results = []
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page   = doc.metadata.get("page", "?")
            snippet = doc.page_content.strip()[:500]
            results.append(f"[Result {i}] ({source} | p.{page})\n{snippet}")
        return "\n\n".join(results)
    except Exception as e:
        return f"Knowledge base search failed: {e}"


# ---------------------------------------------------------------------------
# Tool 2 — Calculator
# ---------------------------------------------------------------------------
_SAFE_PATTERN = re.compile(r"^[\d\s\+\-\*\/\(\)\.\%\*\*]+$")

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the numeric result.
    Supports +, -, *, /, **, % and parentheses. Input MUST be a valid
    Python arithmetic expression (e.g. '123 * 456 + 789', '(10 + 5) ** 2').
    Do NOT use this for anything other than pure arithmetic.
    """
    expression = expression.strip()
    # Strip surrounding quotes the LLM sometimes adds to Action Input
    # e.g.  '2 + 2'  →  2 + 2
    if (expression.startswith("'") and expression.endswith("'")) or \
       (expression.startswith('"') and expression.endswith('"')):
        expression = expression[1:-1].strip()

    if not _SAFE_PATTERN.match(expression):
        return (
            "Error: Only arithmetic expressions are allowed "
            "(digits, +, -, *, /, **, %, parentheses)."
        )
    try:
        result = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"


# ---------------------------------------------------------------------------
# Tool 3 — Current Date & Time
# ---------------------------------------------------------------------------
@tool
def get_current_datetime(unused: str = "") -> str:
    """
    Return the current date and time. Use this when the user asks what
    time or date it is right now. No input is required; pass an empty
    string if the framework requires an argument.
    """
    now = datetime.datetime.now()
    return now.strftime("Current date: %A, %B %d, %Y | Current time: %I:%M %p")


# ---------------------------------------------------------------------------
# Tool 4 — Word Count
# ---------------------------------------------------------------------------
@tool
def word_count(text: str) -> str:
    """
    Count the number of words, characters, and sentences in the provided
    text. Input should be the text you want to analyse. Use this when the
    user asks about statistics or metrics of a piece of text.
    """
    if not text.strip():
        return "Please provide some text to count."
    words      = len(text.split())
    chars      = len(text)
    chars_nsp  = len(text.replace(" ", ""))
    sentences  = len(re.findall(r'[.!?]+', text)) or 1
    return (
        f"Words: {words} | "
        f"Characters (with spaces): {chars} | "
        f"Characters (no spaces): {chars_nsp} | "
        f"Sentences: {sentences}"
    )


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------
ALL_TOOLS = [
    knowledge_base_search,
    calculator,
    get_current_datetime,
    word_count,
]

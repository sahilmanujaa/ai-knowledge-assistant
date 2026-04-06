"""
11 — Agentic AI Demo  (ReAct Prompting)
========================================

Technique: ReAct  (Reason + Act)

High-Level Flow:
─────────────────────────────────────────────────────
  User Question
        ↓
  LLM reads the ReAct prompt
  [Prompt contains every tool's name + docstring]
  [LLM matches the question's intent to the best-fit tool description]
        ↓
  LLM decides next action  ← TOOL SELECTION HAPPENS HERE
  [Outputs structured text:]

    Thought: I need to calculate something → use calculator
    Thought: I need a fact from docs    → use knowledge_base_search
    Thought: I need current time        → use get_current_datetime
    Thought: I need text statistics     → use word_count

  [Writes: Action: <tool_name>  and  Action Input: <input>]
        ↓
  Execute Tool
  [AgentExecutor parses Action/Action Input strings]
  [Looks up the tool by name in ALL_TOOLS list]
  [Calls the Python function with the extracted input]
        ↓
  Return Observation
  [Tool's return string appended to the prompt as Observation:]
        ↓
  LLM decides next action … (loop continues)
        ↓
  Final Answer
─────────────────────────────────────────────────────

Tool Selection — How the LLM Chooses:
──────────────────────────────────────
The LLM does NOT have a classifier or special routing logic.
It reads tool descriptions (docstrings) and uses language understanding:

  Query: "What is 5 * 100?"
  → Reads: calculator docstring says "Evaluate a mathematical expression"
  → Writes: Action: calculator | Action Input: 5 * 100

  Query: "What does the doc say about decorators?"
  → Reads: knowledge_base_search says "Use when you need factual information
           from ingested PDF documents"
  → Writes: Action: knowledge_base_search | Action Input: Python decorators

  Query: "What time is it?"
  → Reads: get_current_datetime says "Use when the user asks what time it is"
  → Writes: Action: get_current_datetime | Action Input: (empty)

Key insight: The docstring IS the tool selection criteria.
Better docstrings = more accurate tool selection by the LLM.

What the LLM generates internally:

  Thought:  I need to search the documentation for this.
  Action:   knowledge_base_search
  Action Input: "LangChain memory module"
  Observation: Retrieved document explaining conversation buffer memory.
  Thought:  I now know the final answer.
  Final Answer: LangChain memory is handled by the Memory module.

Available Tools:
  1. knowledge_base_search  — ChromaDB vector search on ingested PDFs
  2. calculator             — Evaluates arithmetic expressions
  3. get_current_datetime   — Returns current date and time
  4. word_count             — Word / character / sentence statistics

Run:
    python src/agentic_demo/main.py   (from app/ directory)

Demo Queries to Try:
  • "What does the knowledge base say about Python decorators?"
  • "What is 47 * 83 + 256?"
  • "What is today's date and time?"
  • "Count the words in: The quick brown fox jumps over the lazy dog"
  • "Search for Docker networking and also tell me today's date"   ← multi-tool
"""

import os
import sys
import logging

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Ensure `src/` is on sys.path so `agentic_demo` package is importable
# when running as: python src/agentic_demo/main.py  (from the app/ directory)
# ---------------------------------------------------------------------------
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agentic_demo.agent import build_agent   # noqa: E402

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Suppress noisy lower-level logs; keep LangChain agent output visible
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "=" * 70

BANNER = f"""
{DIVIDER}
  🤖  Agentic AI Demo  —  ReAct (Reason + Act) Loop
{DIVIDER}
  Available tools:
    • knowledge_base_search   — search ingested PDF documents
    • calculator              — evaluate arithmetic expressions
    • get_current_datetime    — get current date & time
    • word_count              — analyse text statistics

  Watch the agent's inner monologue:
    Thought → Action → Observation → Thought → … → Final Answer

  Type 'quit', 'exit', or 'q' to stop.
{DIVIDER}
"""

DEMO_QUERIES = [
    "What does the knowledge base say about Python decorators?",
    "What is 47 * 83 + 256?",
    "What is today's date and time?",
    'Count the words in: "The quick brown fox jumps over the lazy dog"',
    "Search for Docker networking and also tell me today's date",
]


def print_intermediate_steps(steps: list) -> None:
    """Pretty-print the tool calls the agent made."""
    if not steps:
        return
    print("\n📋 Agent Steps Summary:")
    for i, (action, observation) in enumerate(steps, 1):
        print(f"  Step {i}: [{action.tool}]  Input → {str(action.tool_input)[:80]}")
        print(f"           Observation → {str(observation)[:120]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY not found. Please set it in your .env file.")
        sys.exit(1)

    print(BANNER)

    # Show demo query suggestions
    print("💡 Demo queries to try:")
    for i, q in enumerate(DEMO_QUERIES, 1):
        print(f"   {i}. {q}")
    print()

    # Build the ReAct agent
    print("⚙️  Building ReAct agent...")
    try:
        executor = build_agent(openai_api_key=OPENAI_API_KEY)
        print("✅ Agent ready.\n")
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        sys.exit(1)

    # ── REPL Loop ────────────────────────────────────────────────────────
    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not query:
            continue

        if query.lower() in {"quit", "exit", "q"}:
            print("Goodbye! 👋")
            break

        print(f"\n{DIVIDER}")
        print(f"❓ Question: {query}")
        print(f"{DIVIDER}\n")

        try:
            result = executor.invoke({"input": query})
        except Exception as e:
            print(f"\n❌ Agent error: {e}")
            print(DIVIDER + "\n")
            continue

        # Print intermediate steps (compact summary after verbose output)
        intermediate = result.get("intermediate_steps", [])
        print_intermediate_steps(intermediate)

        print(f"\n{DIVIDER}")
        print(f"🤖 Final Answer:\n\n{result['output']}")
        print(f"{DIVIDER}\n")


if __name__ == "__main__":
    main()

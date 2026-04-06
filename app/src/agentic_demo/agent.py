"""
agent.py — ReAct Agent Builder
================================

Constructs a LangChain ReAct agent that follows the canonical
Thought → Action → Observation loop until it reaches a Final Answer.

What happens at runtime
-----------------------
1. User question is injected into the ReAct prompt.
2. LLM generates a "Thought" and picks an "Action" (tool name) with
   an "Action Input".
3. AgentExecutor calls the matching tool, captures the output as an
   "Observation", and feeds it back into the prompt.
4. The loop repeats until the LLM writes "Final Answer: …".

The full intermediate reasoning is printed to the terminal because
AgentExecutor is initialised with verbose=True.
"""

import os, sys

# Ensure `src/` is on the path so `agentic_demo.tools` is importable
# whether this file is run directly or via `python -m`.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

from agentic_demo.tools import ALL_TOOLS

# ---------------------------------------------------------------------------
# ReAct Prompt Template
# ---------------------------------------------------------------------------
# This is the standard ReAct prompt (hwchase17/react) reproduced locally
# so the demo works without a LangChain Hub internet call.
# Required input variables: {tools}, {tool_names}, {input}, {agent_scratchpad}
# ---------------------------------------------------------------------------

REACT_TEMPLATE = """You are a helpful AI assistant with access to the following tools:

{tools}

Use the following format EXACTLY — do not deviate:

Question: the input question you must answer
Thought: think step-by-step about what to do
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (Thought / Action / Action Input / Observation can repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

REACT_PROMPT = PromptTemplate.from_template(REACT_TEMPLATE)


# ---------------------------------------------------------------------------
# build_agent — public factory function
# ---------------------------------------------------------------------------

def build_agent(openai_api_key: str, model: str = "gpt-4o-mini") -> AgentExecutor:
    """
    Build and return a ready-to-use AgentExecutor.

    Parameters
    ----------
    openai_api_key : str
        OpenAI API key loaded from .env.
    model : str
        OpenAI chat model to use (default: gpt-4o-mini).

    Returns
    -------
    AgentExecutor
        The executor that drives the Thought → Action → Observation loop.
    """
    llm = ChatOpenAI(
        model=model,
        temperature=0,          # deterministic reasoning
        api_key=openai_api_key,
    )

    # create_react_agent wires: llm + tools + prompt → a runnable agent
    react_agent = create_react_agent(
        llm=llm,
        tools=ALL_TOOLS,
        prompt=REACT_PROMPT,
    )

    executor = AgentExecutor(
        agent=react_agent,
        tools=ALL_TOOLS,
        verbose=True,           # prints every Thought / Action / Observation
        handle_parsing_errors=True,   # gracefully recover from LLM format mistakes
        max_iterations=5,       # safety cap to prevent infinite loops
        return_intermediate_steps=True,
    )

    return executor

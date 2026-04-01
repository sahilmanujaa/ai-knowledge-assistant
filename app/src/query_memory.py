"""
Day 3 — Conversational RAG with Memory
==========================================

Technique: ConversationBufferMemory → Multi-turn Retrieval QA

High-Level Flow:
─────────────────────────────────────────────────────
  User Question  (turn N)
        ↓
  ConversationBufferMemory
  [Appends all previous (question, answer) pairs]
  [Sent as chat_history to the chain automatically]
        ↓
  HuggingFaceEmbeddings  (all-MiniLM-L6-v2)
  + Chroma  —  MMR retriever  (k=3, fetch_k=10)
  [Retrieves diverse chunks relevant to current turn]
        ↓
  ConversationalRetrievalChain
  [Condenses chat_history + new question → standalone query]
  [Runs retrieval on condensed query]
  [Feeds context + history into the LLM]
        ↓
  ChatOpenAI  (gpt-4o-mini)
  [Generates context-aware answer using full conversation]
        ↓
  Answer stored back → Memory  (loop continues)
─────────────────────────────────────────────────────

Key difference from query.py:
  The LLM can refer back to earlier answers in the same session.

Run:
    python src/query_memory.py
"""

import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables (.env)
load_dotenv()

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def create_chain():
    print("Loading embeddings model...")

    embedding_function = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    print("Loading Chroma database...")

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10
        }
    )

    print("Initializing LLM...")

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

    print("Creating conversational chain...")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain


def main():
    print("Starting AI Knowledge Assistant...\n")

    qa_chain = create_chain()

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        result = qa_chain.invoke({"question": query})

        print("\nAnswer:")
        print(result["answer"])

        # print("\nSources:")
        # for doc in result["source_documents"]:
        #     print(doc.metadata)


if __name__ == "__main__":
    main()

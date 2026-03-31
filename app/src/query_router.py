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
DATA_DIR = os.path.join(BASE_DIR, "data")   # app/data/
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────────
# Step 1 — Router LLM + Prompt
# ─────────────────────────────────────────────

router_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

ROUTER_PROMPT = """
You are a query router.

Your job is to decide which knowledge base should answer the question.

Possible sources:
- python
- mysql
- docker

Return ONLY the source name.

Question: {question}
"""

def route_query(question: str) -> str:
    """Routes the question to the appropriate knowledge base name."""
    response = router_llm.invoke(
        [HumanMessage(content=ROUTER_PROMPT.format(question=question))]
    )
    return response.content.strip().lower()


# ─────────────────────────────────────────────
# Step 2 — Load Multiple Vector Databases
# ─────────────────────────────────────────────
# Existing folders inside app/data/:
#
#   chroma_python  → Python docs
#   chroma_mysql   → MySQL / database docs
#   chroma_docker  → Docker / DevOps docs


def create_retrievers(embedding_function):
    """Load all Chroma DBs from app/data/ and return a retriever_map dict."""
    print("Loading Chroma databases from app/data/...")

    python_db = Chroma(
        persist_directory=os.path.join(DATA_DIR, "chroma_python"),
        embedding_function=embedding_function
    )
    mysql_db = Chroma(
        persist_directory=os.path.join(DATA_DIR, "chroma_mysql"),
        embedding_function=embedding_function
    )
    docker_db = Chroma(
        persist_directory=os.path.join(DATA_DIR, "chroma_docker"),
        embedding_function=embedding_function
    )

    # Step 3 — Create retrievers and store in a dictionary
    python_retriever = python_db.as_retriever(search_kwargs={"k": 4})
    mysql_retriever  = mysql_db.as_retriever(search_kwargs={"k": 4})
    docker_retriever = docker_db.as_retriever(search_kwargs={"k": 4})

    retriever_map = {
        "python": python_retriever,
        "mysql":  mysql_retriever,
        "docker": docker_retriever,
    }

    return retriever_map


# ─────────────────────────────────────────────
# Step 4 — Main Chat Loop with Router
# ─────────────────────────────────────────────

def main():
    print("Starting Routed AI Knowledge Assistant...\n")

    print("Loading embeddings model...")
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    retriever_map = create_retrievers(embedding_function)

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

    while True:
        query = input("\nAsk a question (or type 'exit'): ")

        if query.lower() == "exit":
            print("Goodbye!")
            break

        # Route the query to the correct knowledge base
        source = route_query(query)
        print(f"Query routed to: {source}")

        # Fallback to 'python' if the routed source isn't in the map
        retriever = retriever_map.get(source, retriever_map["python"])

        # Build the QA chain with the routed retriever
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )

        result = qa_chain.invoke({"question": query})

        print("\nAnswer:")
        print(result["answer"])

        # Uncomment to show source documents:
        # print("\nSources:")
        # for doc in result["source_documents"]:
        #     print(doc.metadata)


if __name__ == "__main__":
    main()

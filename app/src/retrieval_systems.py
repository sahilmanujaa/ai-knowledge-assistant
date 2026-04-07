import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.base import AttributeInfo
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
INITIAL_K = 20
FINAL_K = 4
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VALID_TOPICS = {"python", "mysql", "docker", "general"}

ROUTER_PROMPT = """You are a query router.

Choose the best topic for the question from:
- python
- mysql
- docker

Return ONLY the topic name.

Question: {question}
"""


def build_llm():
    if not OPENAI_API_KEY:
        return None
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)


def build_vectorstore():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding,
    )
    return embedding, vectorstore


def load_chroma_documents(vectorstore: Chroma):
    data = vectorstore.get()
    docs = []

    if data and data.get("documents"):
        for text, metadata in zip(data["documents"], data["metadatas"]):
            docs.append(Document(page_content=text, metadata=metadata))

    return docs


def build_base_retriever(vectorstore: Chroma):
    return vectorstore.as_retriever(search_kwargs={"k": 3})


def build_multi_query_retriever(vectorstore: Chroma, llm: ChatOpenAI):
    if not llm:
        raise ValueError("Multi-query retrieval requires OPENAI_API_KEY.")

    base_retriever = build_base_retriever(vectorstore)
    return MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)


def build_hybrid_retriever(vectorstore: Chroma):
    docs = load_chroma_documents(vectorstore)
    if not docs:
        return None

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10},
    )
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5],
    )


def build_self_query_retriever(vectorstore: Chroma, llm: ChatOpenAI):
    if not llm:
        raise ValueError("Self-query retrieval requires OPENAI_API_KEY.")

    metadata_field_info = [
        AttributeInfo(
            name="topic",
            description="Topic of the document. Valid values: python, mysql, docker, general",
            type="string",
        ),
        AttributeInfo(
            name="file",
            description="Original name of the PDF file",
            type="string",
        ),
    ]

    return SelfQueryRetriever.from_llm(
        llm,
        vectorstore,
        "Technical documents about python, mysql, and docker",
        metadata_field_info,
    )


def route_topic(question: str, llm: ChatOpenAI) -> str:
    response = llm.invoke([HumanMessage(content=ROUTER_PROMPT.format(question=question))])
    topic = response.content.strip().lower()
    return topic if topic in VALID_TOPICS else "general"


def build_rerank_retriever(vectorstore: Chroma, llm: ChatOpenAI):
    if not llm:
        raise ValueError("Rerank retrieval requires OPENAI_API_KEY.")

    reranker = CrossEncoder(RERANKER_MODEL)

    def retrieve(question: str):
        topic = route_topic(question, llm)
        candidates = vectorstore.similarity_search(
            query=question,
            k=INITIAL_K,
            filter={"topic": topic},
        )

        if not candidates:
            return []

        pairs = [(question, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs).tolist()
        scored_docs = sorted(
            zip(scores, candidates),
            key=lambda item: item[0],
            reverse=True,
        )
        return [doc for _, doc in scored_docs[:FINAL_K]]

    return retrieve

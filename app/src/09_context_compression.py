"""
9 — Context Compression Retrieval Pipeline
======================================================

Technique: Context Compression (LLM-based Extraction)

High-Level Flow:
─────────────────────────────────────────────────────
  User Question
        ↓
  Base Retriever (Chroma Vector Store)
  [Fetches top K somewhat relevant but large/noisy documents]
        ↓
  Document Compressor (LLMChainExtractor)
  [Uses an LLM to read each retrieved document]
  [Extracts ONLY the sentences strictly relevant to the question]
  [Discards irrelevant noise, making the context dense & concise]
        ↓
  Contextual Compression Retriever
  [Returns shortened, highly relevant compressed documents]
        ↓
  ChatPromptTemplate + LLM (gpt-4o-mini)
        ↓
  Grounded, Context-Rich Answer (Lower token usage & less noise)
─────────────────────────────────────────────────────

Use Cases:
----------
1. Long-Winded Documents: When retrieving large chunks where the answer occupies just 5% of the text. Compression extracts the needle and removes the 95% haystack noise, reducing hallucination risk.
2. Token Optimization: Passing large retrieved text wastes LLM context windows and increases API costs. Compression shrinks the prompt size significantly.
3. Multi-Document Synthesis: When you need to retrieve from 20 different documents but pass them to an LLM. Without compression, 20 large chunks won't fit the context window. With compression, you only pass the 20 relevant sentences.

Run:
    python src/09_context_compression.py
"""

import os
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

PROMPT_TEMPLATE = """
Answer the question based only on the following context. 

{context}

---

Answer the question based on the above context: {question}
"""

def init_retriever():
    print("Setting up Contextual Compression Retriever...\n")
    
    # 1. LLM Setup (This will be used both for compressing text AND the final answer)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    # 2. Base Vector Store & Base Retriever
    # Using the standard ingest database we created in script 01
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding)
    
    # Let's fetch a slightly higher number of documents initially
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # 3. Document Compressor (LLM-based extractor)
    print("Initializing LLMChainExtractor (Document Compressor)...")
    compressor = LLMChainExtractor.from_llm(llm)

    # 4. Contextual Compression Retriever Wrap
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever, llm

def main():
    try:
        retriever, answer_llm = init_retriever()
    except Exception as e:
        print(f"Failed to initialize retriever: {e}")
        print("Note: Ensure you have populated the baseline chroma_db using 01_ingest.py!")
        return

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    print("Context Compression Retriever is ready.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        query = input("Enter your query: ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Exiting...")
            break
            
        if not query:
            continue

        print(f"\n[1] Retrieving base documents and applying LLM compression for: '{query}'...")
        print("(This might take slightly longer because the LLM is actively reading and extracting relevant snippets from all fetched docs)")
        
        # This calls the base retriever first, then loops the retrieved docs through the LLM chain extractor
        compressed_docs = retriever.invoke(query)

        if not compressed_docs:
            print("No relevant context found after compression.")
            continue
            
        print(f"Found and compressed {len(compressed_docs)} relevant segments.")
        
        # Build context
        context_text = "\n\n---\n\n".join(
            [doc.page_content for doc in compressed_docs]
        )
        
        print("\n[2] Generating final unified answer...")
        
        # Generate Answer
        prompt = prompt_template.format(context=context_text, question=query)
        response = answer_llm.invoke(prompt)

        print("\n🤖 AI Answer:\n")
        print(response.content)

        print("\n📚 Sources (Compressed Extractions):\n")
        for i, doc in enumerate(compressed_docs):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "Unknown")
            
            print(f"--- Extraction {i+1} ({os.path.basename(source)} | Page {page}) ---")
            print(doc.page_content.strip())
            print()
            
        print("=" * 70 + "\n")

if __name__ == "__main__":
    main()

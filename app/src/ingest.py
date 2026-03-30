import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(base_dir, "data")
CHROMA_PATH = os.path.join(base_dir, "chroma_db")

def load_documents():
    print(f"Loading PDFs from {DATA_PATH}...")
    # Load all PDFs from the data directory
    loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")
    return documents

def split_documents(documents):
    # Split text into chunks of 800 characters, with 80 character overlap
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def save_to_chroma(chunks):
    print("Initializing embedding model (all-MiniLM-L6-v2) and saving to Chroma...")
    # Use the free, local embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Save the chunks to the persistent ChromaDB collection
    db = Chroma.from_documents(
        chunks, 
        embedding_function, 
        persist_directory=CHROMA_PATH
    )
    print(f"Successfully saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    # Make sure data path exists
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created '{DATA_PATH}' directory. Please add '.pdf' files there and run again.")
        return

    documents = load_documents()
    if not documents:
        print(f"No PDF files found in '{DATA_PATH}'. Please add some and run again.")
        return
        
    chunks = split_documents(documents)
    save_to_chroma(chunks)

if __name__ == "__main__":
    main()

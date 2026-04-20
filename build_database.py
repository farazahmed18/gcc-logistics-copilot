import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. Load the environment variables (Your Grok Key)
load_dotenv()

def build_rag_db():
    print("📚 Loading official UAE and GCC logistics documents from ./rag_data...")
    
    # 2. Read all PDFs in the folder
    loader = PyPDFDirectoryLoader("./rag_data")
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages of complex logistics regulations.")

    # 3. Chunk the dense text into smaller, searchable pieces
    print("✂️ Chunking documents into manageable pieces for the AI...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200 
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} text chunks.")

    # 4. Create Embeddings and save to Chroma DB
    print("🧠 Building the Vector Database... (This might take a minute or two)")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("🚀 SUCCESS! RAG Database built and saved to ./chroma_db.")

if __name__ == "__main__":
    build_rag_db()
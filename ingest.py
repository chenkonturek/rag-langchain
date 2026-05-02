import os
import glob
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 80  # stay under free-tier limit of 100 RPM


def load_documents():
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "**/*.pdf"), recursive=True):
        docs.extend(PyPDFLoader(path).load())
    for path in glob.glob(os.path.join(DOCS_DIR, "**/*.txt"), recursive=True):
        docs.extend(TextLoader(path, encoding="utf-8").load())
    return docs


def main():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print(f"No PDF or TXT files found in '{DOCS_DIR}'. Add documents and re-run.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks from {len(documents)} pages/documents.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    print("Embedding and storing in ChromaDB (batching to respect rate limits)...")
    vectorstore = None
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{-(-len(chunks) // BATCH_SIZE)}: chunks {i + 1}–{min(i + BATCH_SIZE, len(chunks))}")
        if vectorstore is None:
            vectorstore = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_DIR)
        else:
            vectorstore.add_documents(batch)
        if i + BATCH_SIZE < len(chunks):
            time.sleep(61)  # wait for the rate-limit window to reset

    print(f"Done. {len(chunks)} chunks stored in '{CHROMA_DIR}'.")


if __name__ == "__main__":
    main()

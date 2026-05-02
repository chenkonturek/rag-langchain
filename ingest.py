import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR = "./docs"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


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

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("Embedding and storing in ChromaDB...")
    Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR)
    print(f"Done. {len(chunks)} chunks stored in '{CHROMA_DIR}'.")


if __name__ == "__main__":
    main()

# RAG Document Q&A

A document question-answering system using Retrieval-Augmented Generation (RAG) built with LangChain and Google's Gemini model.

## Features

- **Multiple document formats** — PDF and TXT files
- **Semantic search** — ChromaDB vector store with persistent storage across sessions
- **AI-powered answers** — Google Gemini 2.0 Flash with source attribution
- **Intelligent chunking** — Recursive text splitting (chunk size: 1000, overlap: 200)

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your Google API key:

```
GOOGLE_API_KEY=your_api_key_here
```

Get a key at [Google AI Studio](https://aistudio.google.com/app/apikey).

## Usage

**1. Add documents**

Place `.pdf` or `.txt` files in the `docs/` directory (subdirectories are supported).

**2. Ingest documents**

```bash
python ingest.py
```

Run this once, and again whenever you add or change files in `docs/`.

**3. Ask questions**

Single question:
```bash
python main.py "What is the main topic of the documents?"
```

Interactive mode:
```bash
python main.py
```

Example output:
```
Answer:
The document describes...

Sources:
  - docs/report.pdf (page 3)
  - docs/notes.txt
```

## Project Structure

```
rag-langchain/
├── docs/          # Place your PDF and TXT documents here
├── chroma_db/     # Persistent vector store (created after first ingest)
├── ingest.py      # Document ingestion pipeline
├── main.py        # Q&A CLI
└── requirements.txt
```

# RAG Document Q&A

A document question-answering system using Retrieval-Augmented Generation (RAG) built with LangChain and Google's Gemini model.

## Features

- **Multiple document formats** — PDF and TXT files
- **Semantic search** — ChromaDB vector store with persistent storage across sessions
- **AI-powered answers** — Google Gemini 2.0 Flash with source attribution
- **Intelligent chunking** — Recursive text splitting (chunk size: 1000, overlap: 200)
- **Web chatbot UI** — Streamlit interface with chat history and source attribution

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

> **Quota note:** If you see a `429 RESOURCE_EXHAUSTED` error with `limit: 0`, your Google Cloud project has billing enabled but no free-tier quota for `gemini-2.0-flash`. Create a fresh API key in a new AI Studio project (without billing attached) to restore the free tier (15 RPM / 1,500 req/day).

## Usage

**1. Add documents**

Place `.pdf` or `.txt` files in the `docs/` directory (subdirectories are supported).

**2. Ingest documents**

```bash
python ingest.py
```

Run this once, and again whenever you add or change files in `docs/`.

**3. Ask questions**

Web chatbot (recommended):
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with a chat interface and collapsible source citations.

Single question (CLI):
```bash
python main.py "What is the main topic of the documents?"
```

Interactive CLI mode:
```bash
python main.py
```

## Project Structure

```
rag-langchain/
├── docs/            # Place your PDF and TXT documents here
├── chroma_db/       # Persistent vector store (created after first ingest)
├── ingest.py        # Document ingestion pipeline
├── main.py          # Q&A CLI and shared chain builder
├── app.py           # Streamlit web chatbot
├── requirements.txt
└── .env.example     # Copy to .env and add GOOGLE_API_KEY
```

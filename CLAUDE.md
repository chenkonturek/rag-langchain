# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add GOOGLE_API_KEY
```

## Commands

```bash
# Index documents (run once, or after adding/changing files in docs/)
python ingest.py

# Ask a single question
python main.py "Your question here"

# Interactive Q&A session
python main.py

# Web chatbot interface
streamlit run app.py
```

## Architecture

Two-script pipeline with a shared ChromaDB store:

**`ingest.py`** — run offline to build the vector index:
1. Recursively loads `docs/**/*.pdf` (via `PyPDFLoader`) and `docs/**/*.txt` (via `TextLoader`)
2. Splits with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)`
3. Embeds with `GoogleGenerativeAIEmbeddings(model="models/embedding-001")`
4. Persists to `./chroma_db/` via `Chroma.from_documents()`

**`main.py`** — query the persisted store at runtime:
- `build_chain()` loads the existing ChromaDB, creates a retriever (`k=4`), and wires an LCEL chain: `retriever → format_docs → ChatPromptTemplate → ChatGoogleGenerativeAI(gemini-2.0-flash) → StrOutputParser`
- After the chain produces an answer, `retriever.invoke()` is called a second time to collect source metadata (file path + page number for PDFs)
- Supports both single-question CLI mode (`sys.argv`) and an interactive REPL

**Key constraint:** `ingest.py` must be re-run whenever `docs/` changes — `main.py` reads only from the persisted `chroma_db/` and will not pick up new documents automatically.

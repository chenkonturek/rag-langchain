import sys
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

CHROMA_DIR = "./chroma_db"

PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        requests_per_minute=80,
    )
    vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer(question: str):
    chain, retriever = build_chain()
    response = chain.invoke(question)
    sources = retriever.invoke(question)

    print(f"\nAnswer:\n{response}")
    print("\nSources:")
    seen = set()
    for doc in sources:
        src = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        label = f"  - {src}" + (f" (page {page + 1})" if page is not None else "")
        if label not in seen:
            seen.add(label)
            print(label)


def interactive_loop():
    print("RAG Q&A — type your question or 'quit' to exit.\n")
    chain, retriever = build_chain()
    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in {"quit", "exit"}:
            break

        response = chain.invoke(question)
        sources = retriever.invoke(question)

        print(f"\nAnswer:\n{response}")
        print("\nSources:")
        seen = set()
        for doc in sources:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            label = f"  - {src}" + (f" (page {page + 1})" if page is not None else "")
            if label not in seen:
                seen.add(label)
                print(label)
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
    else:
        interactive_loop()

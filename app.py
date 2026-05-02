import streamlit as st
from main import build_chain

st.set_page_config(page_title="RAG Chatbot", page_icon="💬")
st.title("RAG Chatbot")


@st.cache_resource
def get_chain():
    return build_chain()


chain, retriever = get_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                st.markdown(msg["sources"])

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            response = chain.invoke(prompt)
            docs = retriever.invoke(prompt)

        st.markdown(response)

        seen = set()
        source_lines = []
        for doc in docs:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            key = (src, page)
            if key not in seen:
                seen.add(key)
                label = f"- `{src}`" + (f", page {page + 1}" if page is not None else "")
                source_lines.append(label)

        sources_md = "\n".join(source_lines) if source_lines else ""
        if sources_md:
            with st.expander("Sources"):
                st.markdown(sources_md)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": sources_md,
    })

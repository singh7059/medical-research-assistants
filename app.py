import streamlit as st
from utils.retrieval import SimpleVectorStore
from utils.web_search import tavily_search
from models.llm import generate_llm_response

# Initialize vector store
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = SimpleVectorStore()

st.title("🧪 Medical Research Assistant")

# Upload research papers / notes
uploaded_files = st.file_uploader(
    "Upload Research Papers (txt)", type=["txt"], accept_multiple_files=True
)
if uploaded_files:
    for file in uploaded_files:
        text = file.read().decode("utf-8")
        st.session_state.vectorstore.add_documents([text])
    st.success("Documents added to knowledge base ✅")

# Response Mode
mode = st.radio("Response Mode", ["Concise", "Detailed"])

# Query Input
query = st.text_input("Ask your medical research question:")

if query:
    st.write("🔍 Searching knowledge base...")
    docs = st.session_state.vectorstore.search(query)

    # Decide whether to use local docs or web search
    min_docs_required = 1  # you can later change this or use a similarity threshold

    if docs and len(docs) >= min_docs_required and any(d.strip() for d in docs):
        context = " ".join(docs)
        st.info("Answer retrieved from local documents ✅")
    else:
        st.info("Answer retrieved from web search 🌍")
        results = tavily_search(query)
        context = " ".join(results)

    # Build prompt for LLM
    if mode == "Concise":
        prompt = f"Answer concisely in 2-3 sentences.\nContext: {context}\nQuestion: {query}"
    else:
        prompt = f"Give a detailed, well-explained answer with references if possible.\nContext: {context}\nQuestion: {query}"

    # Generate LLM response
    response = generate_llm_response(prompt)
    st.subheader("💡 Answer")
    st.write(response)

import os
import streamlit as st
import pickle
import time
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Page Config ---
st.set_page_config(
    page_title=" AI News Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Sidebar ---
st.sidebar.title("🔗 News Article URLs")
urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_hf.pkl"

# --- Main Placeholder ---
main_placeholder = st.empty()

# --- Load Local Model ---
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)

# --- Function to Ask Local Model ---
def ask_local_model(context, question):
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"
    output = pipe(prompt)[0]["generated_text"]
    return output

# --- Process URLs ---
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=[u for u in urls if u])
    main_placeholder.info("📥 Loading data from URLs...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=0
    )
    main_placeholder.info("✂️ Splitting text into chunks...")
    docs = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.success("✅ FAISS embeddings created successfully!")
    time.sleep(1)

    # Save FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

# --- Ask a Question ---
query = main_placeholder.text_input("💬 Ask a question:")
if query and os.path.exists(file_path):
    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    docs = vectorstore.similarity_search(query, k=3)
    context = " ".join([doc.page_content for doc in docs])

    answer = ask_local_model(context, query)

    # --- Display Answer and Sources in Columns ---
    st.markdown("## 📝 Answer")
    st.markdown(f"<div style='background-color:#f0f2f6; padding:15px; border-radius:10px'>{answer}</div>", unsafe_allow_html=True)

    st.markdown("## 🌐 Sources")
    for doc in docs:
        st.markdown(
            f"<div style='background-color:#e3f2fd; padding:10px; border-radius:8px; margin-bottom:5px'>{doc.metadata.get('source','Unknown Source')}</div>",
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown("""
<hr style='border:1px solid #ddd'>
<p style='text-align:center; font-size:12px'>Powered by Streamlit, LangChain & HuggingFace</p>
""", unsafe_allow_html=True)

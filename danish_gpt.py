# app.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
from docx import Document
import io
import re

# -----------------------
# Helpers
# -----------------------
def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text)
    except Exception as e:
        return ""

def extract_text_from_docx(uploaded_file):
    try:
        doc = Document(uploaded_file)
        text = []
        for para in doc.paragraphs:
            text.append(para.text)
        return "\n".join(text)
    except Exception as e:
        return ""

def clean_text(t):
    t = re.sub(r'\s+', ' ', t).strip()
    return t

def chunk_text(text, max_chars=1000, overlap=200):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# -----------------------
# Model loading (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models():
    # embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    # generator: flan-t5-small (instruction-following)
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    # Use CPU (Streamlit free) - if GPU available it will use it automatically
    gen_model.to(torch.device("cpu"))
    return embedder, tokenizer, gen_model

embedder, tokenizer, gen_model = load_models()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Danish-GPT — LinkedIn Chatbot", layout="centered")

st.title("🇩🇰 Danish-GPT — Your LinkedIn chatbot (Streamlit, free)")
st.caption("Upload your CV / LinkedIn text, index it, then recruiters can ask questions. No API keys.")

with st.expander("Quick instructions"):
    st.write("""
    1. Upload your CV (PDF / DOCX / TXT) and/or paste LinkedIn text.  
    2. Click **Index documents** (builds embeddings locally).  
    3. Ask questions in the chat input — responses are generated only from your uploaded content.  
    4. Deploy this app on Streamlit Cloud and add the URL to your LinkedIn Featured section.
    """)

# Sidebar: uploads and settings
st.sidebar.header("Upload & Index (one-time)")
uploaded_cv = st.sidebar.file_uploader("Upload CV (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"])
uploaded_linkedin = st.sidebar.file_uploader("Upload LinkedIn export (PDF / TXT) (optional)", type=["pdf", "txt", "docx"])
pasted_text = st.sidebar.text_area("Or paste LinkedIn / additional notes here", height=120)

max_chars = st.sidebar.slider("Chunk size (chars)", 600, 2000, 1000, step=100)
overlap = st.sidebar.slider("Chunk overlap (chars)", 50, 500, 200, step=50)
top_k = st.sidebar.slider("Context chunks to retrieve (top K)", 1, 6, 3)
st.sidebar.markdown("---")
if "indexed" not in st.session_state:
    st.session_state.indexed = False

if st.sidebar.button("Index documents"):
    # assemble text
    full_text_parts = []

    if uploaded_cv:
        fname = uploaded_cv.name.lower()
        if fname.endswith(".pdf"):
            text = extract_text_from_pdf(uploaded_cv)
        elif fname.endswith(".docx"):
            uploaded_cv.seek(0)
            text = extract_text_from_docx(uploaded_cv)
        else:
            uploaded_cv.seek(0)
            text = uploaded_cv.read().decode("utf-8", errors="ignore")
        full_text_parts.append(text)

    if uploaded_linkedin:
        fname = uploaded_linkedin.name.lower()
        if fname.endswith(".pdf"):
            uploaded_linkedin.seek(0)
            text = extract_text_from_pdf(uploaded_linkedin)
        elif fname.endswith(".docx"):
            uploaded_linkedin.seek(0)
            text = extract_text_from_docx(uploaded_linkedin)
        else:
            uploaded_linkedin.seek(0)
            text = uploaded_linkedin.read().decode("utf-8", errors="ignore")
        full_text_parts.append(text)

    if pasted_text:
        full_text_parts.append(pasted_text)

    full_text = "\n\n".join([clean_text(p) for p in full_text_parts if p and clean_text(p)])
    if not full_text:
        st.sidebar.error("No text found. Upload a PDF/TXT/DOCX or paste text in the sidebar.")
    else:
        with st.spinner("Chunking and embedding... (this may take 20–60s depending on size)"):
            chunks = chunk_text(full_text, max_chars=max_chars, overlap=overlap)
            # embed
            embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
            # store
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.indexed = True
            st.sidebar.success(f"Indexed {len(chunks)} chunks.")

# Chat area
if "history" not in st.session_state:
    st.session_state.history = []

if not st.session_state.indexed:
    st.info("You must index documents first (sidebar). Upload your CV / paste LinkedIn text and click 'Index documents'.")
    st.stop()

st.header("Chat with Danish-GPT")
chat_container = st.container()

# show previous messages
def render_history():
    for message in st.session_state.history:
        role, text = message
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Danish-GPT:** {text}")

render_history()

# input
user_question = st.chat_input("Ask something about my profile / experience...")

if user_question:
    st.session_state.history.append(("user", user_question))
    # compute query embedding
    q_emb = embedder.encode([user_question], convert_to_numpy=True)[0]
    # compute similarities
    sims = cosine_similarity([q_emb], st.session_state.embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    retrieved_chunks = [st.session_state.chunks[i] for i in top_idx]
    # build prompt
    context_text = "\n\n---\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(retrieved_chunks)])
    prompt = (
        "You are Danish-GPT, an assistant that must answer ONLY using the provided context below. "
        "If the answer is not contained in the context, be honest and say you don't know. Keep answers concise and professional.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {user_question}\n\nANSWER:"
    )
    # tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    with st.spinner("Generating answer..."):
        outputs = gen_model.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     max_new_tokens=200,
                                     do_sample=False,
                                     num_beams=2)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # append to history
    st.session_state.history.append(("assistant", answer))

    # re-render chat
    st.experimental_rerun()

# show context used (collapsed)
with st.expander("Show which context chunks were used (transparency)"):
    if "chunks" in st.session_state:
        for i, c in enumerate(st.session_state.chunks[:10]):
            st.write(f"Chunk {i+1}: {c[:400]}{'...' if len(c)>400 else ''}")
    else:
        st.write("No chunks indexed yet.")

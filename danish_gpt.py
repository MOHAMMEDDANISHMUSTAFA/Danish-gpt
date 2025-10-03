# danish_gpt.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# UI Styling
st.set_page_config(page_title="Danish-GPT", layout="centered")
st.markdown("""
    <style>
    body, .main {
        background: linear-gradient(to right, #fce7f3, #fbcfe8);
        color: #1F2937;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #BE185D;
        text-align: center;
        font-size: 48px;
        margin-bottom: 0;
    }
    p.subtitle {
        text-align: center;
        font-size: 20px;
        color: #6B7280;
        margin-top: 0;
    }
    .question-button {
        background-color: #DB2777;
        color: white;
        border: none;
        padding: 12px 20px;
        margin: 10px;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
    }
    .question-button:hover {
        background-color: #F472B6;
    }
    .input-box {
        text-align: center;
        margin-top: 40px;
    }
    .input-box input {
        width: 70%;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        background-color: #FCE7F3;
        color: #1F2937;
    }
    .submit-button {
        background-color: #EC4899;
        border: none;
        padding: 12px;
        border-radius: 50%;
        font-size: 18px;
        color: white;
        margin-left: 10px;
        cursor: pointer;
    }
    .response-box {
        margin-top: 40px;
        background-color: #F9A8D4;
        padding: 20px;
        border-radius: 10px;
        color: #1F2937;
    }
    .highlight {
        font-weight: bold;
        color: #BE185D;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>Welcome to Danish-GPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask anything about Danish's projects, experience, and goals.</p>", unsafe_allow_html=True)

# CV Content
profile_text = """[PASTE YOUR FULL CV TEXT HERE — SAME AS BEFORE]"""

# Load Models
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_model.to(torch.device("cpu"))
    return embedder, tokenizer, gen_model

embedder, tokenizer, gen_model = load_models()

def clean_text(t): return re.sub(r'\s+', ' ', t).strip()
def chunk_text(text, max_chars=1000, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks

chunks = chunk_text(clean_text(profile_text))
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# Preloaded Buttons
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
cols = st.columns(4)
with cols[0]: 
    if st.button("What are Danish's top projects?"): 
        st.session_state.question = "What are Danish's top projects?"
with cols[1]: 
    if st.button("Summarize Danish's recent experience"): 
        st.session_state.question = "Summarize Danish's recent experience"
with cols[2]: 
    if st.button("What roles is Danish targeting now?"): 
        st.session_state.question = "What roles is Danish targeting now?"
with cols[3]: 
    if st.button("Who is Danish?"): 
        st.session_state.question = "Who is Danish?"
st.markdown("</div>", unsafe_allow_html=True)

# Input Box
st.markdown("<div class='input-box'>", unsafe_allow_html=True)
col_input, col_button = st.columns([0.85, 0.15])
with col_input:
    user_input = st.text_input("Want to know more about Danish? Ask away...", key="main_input")
with col_button:
    if st.button("❤️", key="submit", help="Submit your question"):
        st.session_state.question = user_input
st.markdown("</div>", unsafe_allow_html=True)

# Answer Generator
def generate_answer(question):
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:3]
    context = "\n\n".join([chunks[i] for i in top_idx])
    prompt = f"You are Danish-GPT, an assistant that answers ONLY using the context below.\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = gen_model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=200, do_sample=False, num_beams=2)
    raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    for kw in ["Generative AI", "NLP", "Big Data", "Data Science", "LangChain", "FastAPI", "Playwright"]:
        raw_answer = raw_answer.replace(kw, f"<span class='highlight'>{kw}</span>")
    return raw_answer, sims[top_idx[0]]

# Display Response + History
query = st.session_state.get("question")
if query:
    answer, score = generate_answer(query)
    if "history" not in st.session_state: st.session_state.history = []
    st.session_state.history.append((query, answer, score))
    st.markdown(f"<div class='response-box'><strong>You asked:</strong> {query}<br><br><strong>Danish-GPT:</strong> {answer}<br><br><em>Confidence score: {score:.2f}</em></div>", unsafe_allow_html=True)

# Show History
if "history" in st.session_state and len(st.session_state.history) > 1:
    st.markdown("### Previous Questions")
    for q, a, s in reversed(st.session_state.history[:-1]):
        st.markdown(f"<div class='response-box'><strong>You asked:</strong> {q}<br><br><strong>Danish-GPT:</strong> {a}<br><br><em>Confidence score: {s:.2f}</em></div>", unsafe_allow_html=True)

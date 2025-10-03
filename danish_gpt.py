# danish_gpt.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# -----------------------
# Custom Styling
# -----------------------
st.markdown("""
    <style>
    body, .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stChatInput textarea {
        background-color: #1E293B;
        color: #F8FAFC;
    }
    .stTextInput, .stTextArea, .stSelectbox, .stSlider {
        background-color: #1E293B !important;
        color: #F8FAFC !important;
    }
    .stButton>button {
        background-color: #14B8A6;
        color: white;
        border-radius: 5px;
    }
    .chat-bubble-user {
        background-color: #1E40AF;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    .chat-bubble-bot {
        background-color: #334155;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Helpers
# -----------------------
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
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_model.to(torch.device("cpu"))
    return embedder, tokenizer, gen_model

embedder, tokenizer, gen_model = load_models()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Danish-GPT", layout="centered")
st.markdown("""
    <h1 style='text-align: center; color: #14B8A6; font-size: 42px; font-weight: bold;'>
        Danish-GPT
    </h1>
    <p style='text-align: center; color: #F8FAFC; font-size: 16px;'>
        Ask anything about Mohammed Danish Mustafa's profile.
    </p>
""", unsafe_allow_html=True)

# Preloaded profile text
profile_text = """
MOHAMMED DANISH MUSTAFA — Data Scientist | AI Engineer | Generative AI
Email: modamu96@gmail.com | Phone: +33 745 579 193 | Paris, France

Professional Summary:
AI & Data Scientist with hands-on experience in NLP pipelines, predictive models, and BI solutions. Skilled in LLMs, RAG, LangChain, and cloud-native deployments. Strong communicator, collaborative mindset, and passion for scalable AI products. Open to full-time roles.

Core Competencies:
LLMs, RAG, LangChain, Transformers, Knowledge Graphs, Generative AI, NLP, Python, SQL, Git, CI/CD, Snowflake, Databricks, Dataiku, Power BI, Tableau, Dash, GCP, Docker, REST APIs, FAISS, Weaviate, Playwright, FastAPI.

Certifications:
Google Cloud GenAI Leader | Neo4j Certified Professional | AWS Cloud Practitioner | AWS Solutions Architect

Experience:
• AI Developer @ SitinCloud (Jan 2025 – Present, Paris): Built LLM apps, RAG pipelines, NLP workflows, knowledge graphs, SEO automation, semantic search crawlers, teaser generation tools.
• Data Analyst @ Modemo (Nov 2023 – Feb 2024, Nantes): Built SQL ETL pipelines, improved forecasts, delivered ML solutions.
• Village Secretary @ Govt of Andhra Pradesh (2019–2022): Managed documentation, budgets, and program implementation.
• Electronics Engineer @ GreenTree (2017–2019): Designed digital communication systems and IoT smart home features.

Education:
• MSc Data Analytics @ DSTI, France (Oct 2023 – Dec 2024)
• Bachelors in Electronics & Communication @ JNTUA, India (2013–2017)

LinkedIn Highlights:
• Passionate AI engineer with expertise in RAG, LangChain, FastAPI, Playwright, vector databases, and automation.
• Built predictive models, NLP pipelines, dashboards, simulations, and SEO tools.
• 1,185 followers | Open to roles in Generative AI, NLP, Big Data, Data Science, and Business Analytics.
"""

# Indexing
max_chars = 1000
overlap = 200
top_k = 3
chunks = chunk_text(clean_text(profile_text), max_chars=max_chars, overlap=overlap)
embeddings = embedder.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
st.session_state.chunks = chunks
st.session_state.embeddings = embeddings
st.session_state.indexed = True

# Chat
if "history" not in st.session_state:
    st.session_state.history = []

st.header("Chat with Danish-GPT")
for role, text in st.session_state.history:
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
    st.markdown(f'<div class="{bubble_class}">{text}</div>', unsafe_allow_html=True)

user_question = st.chat_input("Ask something about my profile / experience...")
if user_question:
    st.session_state.history.append(("user", user_question))
    q_emb = embedder.encode([user_question], convert_to_numpy=True)[0]
    sims = cosine_similarity([q_emb], st.session_state.embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]
    retrieved_chunks = [st.session_state.chunks[i] for i in top_idx]
    context_text = "\n\n---\n\n".join([f"Context {i+1}:\n{c}" for i, c in enumerate(retrieved_chunks)])
    prompt = (
        "You are Danish-GPT, an assistant that must answer ONLY using the provided context below. "
        "If the answer is not contained in the context, be honest and say you don't know. Keep answers concise and professional.\n\n"
        f"CONTEXT:\n{context_text}\n\nQUESTION: {user_question}\n\nANSWER:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = gen_model.generate(input_ids=inputs.input_ids,
                                 attention_mask=inputs.attention_mask,
                                 max_new_tokens=200,
                                 do_sample=False,
                                 num_beams=2)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.session_state.history.append(("assistant", answer))

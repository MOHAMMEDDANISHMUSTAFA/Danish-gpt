# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk

# Ensure punkt is available (download quietly at runtime)
nltk.download("punkt", quiet=True)
from nltk import sent_tokenize

# ---------------------------
# YOUR PRELOADED CV / LINKEDIN
# ---------------------------
DOCUMENTS = """
Mohammed Danish Mustafa 
Data Scientist | AI Engineer | Generative AI 
modamu96@gmail.com | +33 745 579 193 | Paris, France 
Linkedin | Github | Portfolio 

Professional Summary 
AI & Data Scientist with hands-on experience in developing NLP pipelines, predictive models, and business 
intelligence solutions. Skilled in Large Language Models (LLMs), Retrieval Augmented Generation (RAG), and cloud
native deployments. Passionate about building scalable, intelligent systems that solve real-world problems. Strong 
communicator with a collaborative mindset and a drive for continuous learning with a passion for designing scalable 
AI products. Actively seeking a full-time opportunity and available to start immediately. 

Core Competencies 
• AI & ML: LLMs, RAG, Lang chain, Transformers, Knowledge Graphs, Generative AI 
• NLP: Text chunking, classification, sentiment analysis, embeddings 
• Programming: Python (advanced), SQL, Git, CI/CD 
• Data Platforms: Snowflake, Databricks, Dataiku 
• Visualization: Power BI, Tableau 
• Prototyping Tools: Dash framework 
• Cloud & DevOps: GCP, Docker, REST APIs 
• Other: Vectorization strategies, multi-modal data representation 
• Certifications: Google Cloud GenAI Leader | Neo4j Certified Professional 

Professional Experience 
AI Developer, Sitincloud – Paris, France 
Jan 2025 – Present 
• Designed and deployed LLM-based applications using Lang chain and vector databases 
• Implemented RAG pipelines to enhance contextual accuracy in chatbot responses 
• Built Python-based NLP workflows for document classification and sentiment analysis 
• Applied chunking strategies to optimize LLM input processing 
• Created knowledge graphs to support semantic search and entity linking 
• Integrated solutions with Snowflake and Databricks for scalable data handling 
• Delivered insights through Power BI dashboards and custom analytics tools 
• Developed interactive dashboards and prototypes using Dash framework to visualize AI model outputs for 
stakeholders. 

Data Analyst, Modemo – Nantes, France 
Nov 2023 – Jan 2024 
• Built SQL-based ETL pipelines for automated reporting and analytics across finance and operations. 
• Improved forecast accuracy by 18% using time series models. 
• Translated business requirements into production-ready data and ML solutions. 

Education 
• MSc Data Analytics, DSTI (France) – Oct 2023 to Nov 2024 
• Bachelors in Electronics & Communication Engineering, JNTUA(India) – 2013 to 2017

LinkedIn Summary:
Hi, I’m Danish, a passionate AI engineer with a strong background in data analytics and hands-on experience in building intelligent, scalable, and automated systems. I recently completed my Master’s in Data Analytics at Data ScienceTech Institute (DSTI), France, and have worked on impactful AI projects at SitInCloud, focusing on generative AI, automation, and data-driven solutions.

Expertise:
- Developing RAG pipelines for knowledge retrieval and intelligent Q&A systems
- Designing AI agents and modular workflows with LangChain
- Deploying AI microservices with FastAPI
- Implementing web automation with Playwright
- Leveraging vector databases (FAISS, Weaviate) for semantic search
- Automating SEO workflows and creating custom NLP-based solutions

Projects & Highlights:
- SEO Topic Modeling & Analysis (Python, BERTopic, UMAP, ChromaDB) — automated PDF/PowerPoint reports, reduced manual reporting effort by 80%.
- Company Intelligence Enrichment — domain-to-metadata enrichment tool.
- Teaser Generation — AI-powered teaser generation tool (OpenAI API, LangChain).
- Web Crawler & Browser Automation Agent — Playwright, LangChain, Tor.

Other roles: Village Secretary (Govt. of Andhra Pradesh), Electronics Engineer (GreenTree Tech).
Languages: English (Native/bilingual), French (Elementary), Hindi, Urdu.
Certifications: Google Cloud GenAI Leader (Jul 2025), Neo4j Certified Professional (Jul 2024).
"""

# ---------------------------
# Utility: chunking + cleaning
# ---------------------------
def clean_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()

def chunk_text(text: str, max_chars: int = 800, overlap: int = 200):
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(L, start + max_chars)
        chunk = text[start:end].strip()
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

# ---------------------------
# Load embedding model once
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

# ---------------------------
# Preprocess doc: chunks & sentence embeddings
# ---------------------------
@st.cache_data(show_spinner=False)
def index_document(text: str, max_chars=800, overlap=200):
    text = clean_text(text)
    chunks = chunk_text(text, max_chars=max_chars, overlap=overlap)
    # map chunk -> sentences
    sentences = []
    sentence_chunk_map = []  # sentence index -> chunk index
    for i, c in enumerate(chunks):
        sents = sent_tokenize(c)
        for s in sents:
            s_clean = clean_text(s)
            if s_clean:
                sentences.append(s_clean)
                sentence_chunk_map.append(i)
    # embeddings
    chunk_embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sentence_embeddings = embedder.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
    return {
        "chunks": chunks,
        "chunk_embeddings": chunk_embeddings,
        "sentences": sentences,
        "sentence_embeddings": sentence_embeddings,
        "sentence_chunk_map": sentence_chunk_map,
    }

with st.spinner("Indexing Danish's CV and LinkedIn content..."):
    index = index_document(DOCUMENTS, max_chars=800, overlap=200)

# ---------------------------
# Streamlit UI - clean look
# ---------------------------
st.set_page_config(page_title="🇩🇰 Danish-GPT", layout="centered", page_icon="🤖")
st.markdown("<h1 style='margin-bottom:6px'>🇩🇰 Danish-GPT — Ask about Danish</h1>", unsafe_allow_html=True)
st.markdown("Ask me about Danish’s **skills, projects, experience, education, or certifications**. Answers come only from his CV & LinkedIn summary.")

# small helper to render chat bubbles
def user_bubble(text):
    st.markdown(f"<div style='background:#e6f2ff;padding:10px;border-radius:10px;margin:8px 0'><b>You:</b> {text}</div>", unsafe_allow_html=True)

def bot_bubble(text):
    st.markdown(f"<div style='background:#f1f1f1;padding:12px;border-radius:10px;margin:8px 0'><b>Danish-GPT:</b> {text}</div>", unsafe_allow_html=True)

# initialize session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of tuples (role, text, sources_list)

# input form
with st.form(key="ask_form", clear_on_submit=True):
    query = st.text_input("Type your question about Danish (e.g., 'What ML projects has he done?')", max_chars=300)
    submitted = st.form_submit_button("Ask Danish-GPT")

# answering logic (extractive)
def answer_query(query_text: str, top_sentences: int = 5, sim_threshold: float = 0.35):
    q_emb = embedder.encode([query_text], convert_to_numpy=True)[0]
    # sentence-level similarity
    sims = cosine_similarity([q_emb], index["sentence_embeddings"])[0]
    # get top sentence indices
    top_idx = np.argsort(sims)[::-1][:top_sentences]
    top_scores = sims[top_idx]
    # if top score too low, no reliable answer
    if top_scores.size == 0 or top_scores[0] < sim_threshold:
        return "I don't know the answer from Danish's CV/LinkedIn. Try asking about skills, projects, or certifications.", []
    selected_sentences = []
    used_chunks = set()
    for idx, score in zip(top_idx, top_scores):
        if score < sim_threshold:
            continue
        sent = index["sentences"][int(idx)]
        if sent not in selected_sentences:
            selected_sentences.append(sent)
            used_chunks.add(index["sentence_chunk_map"][int(idx)])
    # Join sentences into a concise answer (preserve order as in doc)
    # Sort selected_sentences by their order in the document
    # Find their indices to sort
    ordered = sorted(selected_sentences, key=lambda s: index["sentences"].index(s))
    answer = " ".join(ordered)
    # clamp length
    if len(answer) > 900:
        answer = answer[:900].rsplit(".", 1)[0] + "."
    return answer, sorted(list(used_chunks))

# handle submission
if submitted and query:
    st.session_state.history.append(("user", query, []))
    with st.spinner("Fetching answer..."):
        ans, sources = answer_query(query, top_sentences=6, sim_threshold=0.32)
    st.session_state.history.append(("assistant", ans, sources))

# render chat history
for role, text, sources in st.session_state.history:
    if role == "user":
        user_bubble(text)
    else:
        bot_bubble(text)
        # show sources small
        if sources:
            chunk_preview = []
            for c in sources:
                short = index["chunks"][c][:350]
                chunk_preview.append(f"• Chunk {c+1}: {short}{'...' if len(index['chunks'][c])>350 else ''}")
            with st.expander("Show source chunks used for this answer"):
                for p in chunk_preview:
                    st.markdown(p)

# footer / tips
st.markdown("---")
st.markdown("**Tips:** Try questions like _'What AI tools does Danish use?'_, _'Tell me about his SitinCloud role'_, or _'Which certifications does he have?'_")
st.caption("This assistant answers strictly from the uploaded CV & LinkedIn text. If it can't find an answer, it will say so.")

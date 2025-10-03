# danish_gpt.py
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# -----------------------
# UI Styling
# -----------------------
st.set_page_config(page_title="Danish-GPT", layout="centered")
st.markdown("""
    <style>
    body, .main {
        background-color: #0F172A;
        color: #F8FAFC;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #14B8A6;
        text-align: center;
        font-size: 48px;
        margin-bottom: 0;
    }
    p.subtitle {
        text-align: center;
        font-size: 20px;
        color: #CBD5E1;
        margin-top: 0;
    }
    .question-button {
        background-color: #1E3A8A;
        color: white;
        border: none;
        padding: 12px 20px;
        margin: 10px;
        border-radius: 8px;
        font-size: 16px;
        cursor: pointer;
    }
    .question-button:hover {
        background-color: #2563EB;
    }
    .input-box {
        text-align: center;
        margin-top: 40px;
    }
    .input-box input {
        width: 80%;
        padding: 12px;
        font-size: 16px;
        border-radius: 8px;
        border: none;
        background-color: #1E293B;
        color: #F8FAFC;
    }
    .response-box {
        margin-top: 40px;
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        color: #F8FAFC;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Header
# -----------------------
st.markdown("<h1>Danish-GPT</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask anything about Danish's projects, experience, and goals.</p>", unsafe_allow_html=True)

# -----------------------
# Full CV + Projects
# -----------------------
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

Projects:
SEO Topic Modeling & Analysis — Tools: Python, BERTopic, SentenceTransformer, UMAP, KMeans, ChromaDB, Pandas
Developed an automated SEO topic modeling pipeline to extract, cluster, and visualize search engine content.
Used embeddings and UMAP for semantic understanding; KMeans with silhouette score for clustering.
Generated automated PDF/PPT reports with keyword trends. Reduced manual effort by 80%.

Company Intelligence Enrichment — Tools: Python, APIs, Pandas
Built a domain-to-metadata enrichment tool to gather structured company data without scraping.
Automated extraction of company profiles, industry info, and metrics. Enabled fast insights for BD teams.

Teaser Generation — Tools: Python, OpenAI GPT API, NLP
Created AI-powered teaser generator for marketing-ready summaries from long documents.
Accelerated content creation by 70–80% with high relevance and quality.

Web Crawler & Browser Automation Agent — Tools: Python, Playwright, LangChain, Tor
Built a robust agent to navigate sites, search/download papers, and handle popups/errors.
Integrated GPT to analyze and store relevant content. Enabled large-scale automated research.
"""

# -----------------------
# Embedding Setup
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    gen_model.to(torch.device("cpu"))
    return embedder, tokenizer, gen_model

embedder, tokenizer, gen_model = load_models()

def clean_text(t):
    return re.sub(r'\s+', ' ', t).strip()

def chunk_text(text, max_chars=1000, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end].strip())
        start = end - overlap
    return chunks

chunks = chunk_text(clean_text(profile_text))
embeddings = embedder.encode(chunks, convert_to_numpy=True)

# -----------------------
# Preloaded Questions
# -----------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Top projects"):
        st.session_state.question = "What are Danish's top projects?"
with col2:
    if st.button("Recent experience"):
        st.session_state.question = "Summarize Danish's recent experience"
with col3:
    if st.button("Target roles"):
        st.session_state.question = "What roles is Danish targeting now?"
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Input Box
# -----------------------
st.markdown("<div class='input-box'>", unsafe_allow_html=True)
user_input = st.text_input("Want to know more about Danish? Ask away...", key="main_input")
st.markdown("</div>", unsafe_allow_html=True)

# -----------------------
# Response Logic
# -----------------------
def generate_answer(question):
    q_emb = embedder.encode([question], convert_to_numpy=True)[0]
    sims = cosine_similarity([q_emb], embeddings)[0]
    top_idx = np.argsort(sims)[::-1][:3]
    context = "\n\n".join([chunks[i] for i in top_idx])
    prompt = (
        "You are Danish-GPT, an assistant that answers ONLY using the context below.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = gen_model.generate(input_ids=inputs.input_ids,
                                 attention_mask=inputs.attention_mask,
                                 max_new_tokens=200,
                                 do_sample=False,
                                 num_beams=2)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------
# Display Response
# -----------------------
query = st.session_state.get("question") or user_input
if query:
    answer = generate_answer(query)
    st.markdown(f"<div class='response-box'><strong>You asked:</strong> {query}<br><br><strong>Danish-GPT:</strong> {answer}</div>", unsafe_allow_html=True)

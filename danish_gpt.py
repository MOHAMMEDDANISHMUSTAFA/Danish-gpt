import streamlit as st
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from langchain.llms import HuggingFacePipeline

# Load vector DB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.load_local("danish_vector_db", embedding_model, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever()

# Load model
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
llm = HuggingFacePipeline(pipeline=pipe)

# QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Streamlit UI
st.set_page_config(page_title="Danish-GPT", page_icon="🤖")
st.title("🤖 Danish-GPT")
st.write("Ask me anything about Danish's experience, skills, or projects.")

query = st.text_input("Type your question here:")

if query:
    response = qa_chain.invoke(query)
    st.markdown(f"**🧠 Danish-GPT says:**\n\n{response}")

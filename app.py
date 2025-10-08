# ---------------------- IMPORTS ----------------------
import os
import time
import warnings
import streamlit as st
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from groq import Groq
import uuid
from dotenv import load_dotenv

# ---------------------- ENVIRONMENT SETUP ----------------------
load_dotenv()
warnings.filterwarnings("ignore")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

groq_client = Groq(api_key=GROQ_API_KEY)
nlp = spacy.load("en_core_web_sm")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------- CHROMADB SETUP ----------------------
CHROMA_DB_PATH = "./chroma_db" if os.path.exists("./chroma_db") else None
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DB_PATH
    )
)

try:
    collection = chroma_client.get_or_create_collection(name="constitution")
except Exception:
    # If schema mismatch, reset DB
    if CHROMA_DB_PATH:
        import shutil
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
    chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_PATH)
    )
    collection = chroma_client.get_or_create_collection(name="constitution")

# ---------------------- PDF & KNOWLEDGE BASE ----------------------
PDF_PATH = os.path.join(os.path.dirname(__file__), "COK.pdf")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "".join([page.extract_text() or "" for page in pdf.pages])

def chunk_text(text, max_tokens=500):
    doc = nlp(text)
    chunks, current_chunk, current_length = [], "", 0
    for sent in doc.sents:
        sent_text = sent.text
        sent_tokens = len(nlp(sent_text))
        if current_length + sent_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sent_text
            current_length = sent_tokens
        else:
            current_chunk += " " + sent_text
            current_length += sent_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def embed_and_store(chunks):
    embeddings = embedder.encode(chunks)
    for chunk, embedding in zip(chunks, embeddings):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(uuid.uuid4())]
        )

def query_constitution(query, n_results=5):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=n_results)
    return results['documents'][0]

def generate_response(query, context):
    prompt = f"""
You are a legal assistant specializing in the Kenyan Constitution.
Context:
{context}

Query:
{query}

Answer:
"""
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a knowledgeable legal assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=500
    )
    return response.choices[0].message.content

def setup_knowledge_base():
    if collection.count() == 0:
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        embed_and_store(chunks)

setup_knowledge_base()

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(page_title="BOBA", page_icon="🇰🇪", layout="centered")

st.markdown("""
<style>
body { background-color: #121212; color: white; }
div[data-testid="stTextArea"] textarea {
    background-color: #2a2a2a; color: white;
    border: 2px solid white; border-radius: 5px;
}
.response-box {
    background: #1e1e1e; padding: 15px; border-left: 6px solid red;
    border-radius: 5px; min-height: 100px; white-space: pre-wrap;
    font-family: Consolas, monospace;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
<h1>Kenyan Constitution Chatbot</h1>
</div>
""", unsafe_allow_html=True)

language = st.selectbox("Language:", ["English", "Swahili"])
question = st.text_area("Ask about Kenya's Constitution:")

def chatbot_response(question, lang="en"):
    context = query_constitution(question)
    return generate_response(question, context)

if st.button("Ask", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        placeholder = st.empty()
        placeholder.markdown('<div class="response-box">Thinking...</div>', unsafe_allow_html=True)
        answer = chatbot_response(question, "sw" if language == "Swahili" else "en")
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(f'<div class="response-box">{typed_text}</div>', unsafe_allow_html=True)
            time.sleep(0.015)

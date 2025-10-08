import streamlit as st
import time

# ---------------------- IMPORT YOUR CHAT LOGIC ----------------------
# Importing libraries
import os
import warnings
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
from groq import Groq
import uuid
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")


# Retrieve the GROQ API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize the Groq client for generating responses
groq_client = Groq(api_key=GROQ_API_KEY)

# Load spaCy model for NLP tasks 
nlp = spacy.load("en_core_web_sm")

# Initialize SentenceTransformer for generating text embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize ChromaDB client for persistent vector storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get a collection named "constitution" in ChromaDB
collection = chroma_client.get_or_create_collection(name="constitution")

# Define the path to the Constitution PDF using a relative path
PDF_PATH = os.path.join(os.path.dirname(__file__), "COK.pdf")

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

# Function to chunk text into smaller pieces for embedding
def chunk_text(text, max_tokens=500):
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    current_length = 0
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

# Function to embed text chunks and store them in ChromaDB
def embed_and_store(chunks):
    embeddings = embedder.encode(chunks)
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(uuid.uuid4())]
        )

# Function to query the Constitution knowledge base using a text query
def query_constitution(query, n_results=5):
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results['documents'][0]

# Function to generate a response using Groq API based on the query and context
def generate_response(query, context):
    prompt = f"""
    You are a legal assistant specializing in the Kenyan Constitution. Based on the following context from the Kenyan Constitution, answer the query accurately and concisely. If the context is insufficient, indicate so and provide a general response based on your knowledge.

    Context:
    {context}

    Query:
    {query}

    Answer:
    """
    # Call the Groq API to generate a response
    response = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a knowledgeable legal assistant."},
            {"role": "user", "content": prompt}
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=500
    )
    return response.choices[0].message.content

# Function to set up the knowledge base by processing the Constitution PDF
def setup_knowledge_base():
    if collection.count() == 0:
        print("Processing Kenyan Constitution PDF...")
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        embed_and_store(chunks)
        print("Constitution data processed and stored in ChromaDB.")
    else:
        print("Using existing ChromaDB collection.")
# -------------------------------------------------------------------

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="BOBA", page_icon="🇰🇪", layout="centered")

# ---------------------- CUSTOM CSS ----------------------
st.markdown("""
    <style>
        body { background-color: #121212; color: white; }
        .flag-stripe { height: 8px; width: 100%; }
        .black-stripe { background-color: black; }
        .red-stripe { background-color: red; }
        .white-stripe { background-color: white; }
        .green-stripe { background-color: green; }
        .header {
            display: flex; align-items: center; justify-content: center;
            gap: 15px; margin: 20px 0;
        }
        .header img { height: 50px; }
        .header h1 { color: #ff0000; font-size: 32px; margin: 0; }
        div[data-testid="stTextArea"] textarea {
            background-color: #2a2a2a; color: white;
            border: 2px solid white; border-radius: 5px;
        }
        .response-box {
            background: #1e1e1e; padding: 15px; border-left: 6px solid red;
            border-radius: 5px; min-height: 100px;
            white-space: pre-wrap; font-family: Consolas, monospace;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------- FLAG STRIPES (TOP) ----------------------
st.markdown("""
<div class="flag-stripe black-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe red-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe green-stripe"></div>
""", unsafe_allow_html=True)

# ---------------------- HEADER ----------------------
st.markdown("""
<div class="header">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png" />
    <h1>Kenyan Constitution Chatbot</h1>
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/1200px-Flag_of_Kenya.svg.png" />
</div>
""", unsafe_allow_html=True)

# ---------------------- INPUT CONTROLS ----------------------
language = st.selectbox("Language:", ["English", "Swahili"])
question = st.text_area("Ask about Kenya's Constitution:")

# ---------------------- ASK BUTTON ----------------------
if st.button("Ask", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        placeholder = st.empty()
        placeholder.markdown('<div class="response-box">Thinking...</div>', unsafe_allow_html=True)

        # Call your chatbot logic directly (no HTTP call)
        answer = chatbot_response(question, "sw" if language == "Swahili" else "en")

        # Typewriter animation
        typed_text = ""
        for char in answer:
            typed_text += char
            placeholder.markdown(f'<div class="response-box">{typed_text}</div>', unsafe_allow_html=True)
            time.sleep(0.015)

# ---------------------- FLAG STRIPES (BOTTOM) ----------------------
st.markdown("""
<div class="flag-stripe black-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe red-stripe"></div>
<div class="flag-stripe white-stripe"></div>
<div class="flag-stripe green-stripe"></div>
""", unsafe_allow_html=True)

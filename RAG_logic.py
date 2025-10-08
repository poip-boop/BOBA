# Importing libraries
import os
import warnings
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from groq import Groq
import uuid

# Load environment variables from .env file
load_dotenv()
# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# ------------------ ENVIRONMENT SETUP ------------------ #

# Retrieve the GROQ API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize the Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Initialize SentenceTransformer for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ------------------ CHROMADB SETUP ------------------ #

# Use persistent storage if available, otherwise in-memory (useful for Streamlit Cloud)
CHROMA_DB_PATH = "./chroma_db" if os.path.exists("./chroma_db") else None
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DB_PATH
    )
)

# Create or get collection safely
try:
    collection = chroma_client.get_or_create_collection(name="constitution")
except Exception as e:
    # If schema mismatch occurs, recreate DB
    if CHROMA_DB_PATH:
        import shutil
        shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
    chroma_client = chromadb.Client(
        Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DB_PATH)
    )
    collection = chroma_client.get_or_create_collection(name="constitution")

# ------------------ PDF SETUP ------------------ #

PDF_PATH = os.path.join(os.path.dirname(__file__), "COK.pdf")
if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF file not found at {PDF_PATH}")

# ------------------ FUNCTION DEFINITIONS ------------------ #

def extract_text_from_pdf(pdf_path):
    """Extract text from each page of a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "".join([page.extract_text() or "" for page in pdf.pages])
    return text

def chunk_text(text, max_tokens=500):
    """Split text into chunks of roughly max_tokens using spaCy sentences."""
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

def embed_and_store(chunks):
    """Embed text chunks and store in ChromaDB collection."""
    embeddings = embedder.encode(chunks)
    for chunk, embedding in zip(chunks, embeddings):
        collection.add(
            documents=[chunk],
            embeddings=[embedding.tolist()],
            ids=[str(uuid.uuid4())]
        )

def query_constitution(query, n_results=5):
    """Retrieve relevant documents from ChromaDB based on query."""
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results
    )
    return results['documents'][0]

def generate_response(query, context):
    """Generate a response using Groq LLM based on query and context."""
    prompt = f"""
    You are a legal assistant specializing in the Kenyan Constitution. Based on the following context from the Kenyan Constitution, answer the query accurately and concisely. If the context is insufficient, indicate so and provide a general response based on your knowledge.

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
    """Process PDF and populate ChromaDB if empty."""
    if collection.count() == 0:
        print("Processing Kenyan Constitution PDF...")
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        embed_and_store(chunks)
        print("Constitution data processed and stored in ChromaDB.")
    else:
        print("Using existing ChromaDB collection.")

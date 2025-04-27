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

# Suppress warnings to keep the output clean
warnings.filterwarnings("ignore")

# Loading environment variables 
load_dotenv()

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
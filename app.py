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

# Use st.secrets for deployment, fall back to .env for local
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please configure it in Streamlit secrets.")
    st.stop()

# ---------------------- CACHED RESOURCES ----------------------
@st.cache_resource
def load_models():
    """Load heavy models once and cache them"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("Spacy model not found. Run: python -m spacy download en_core_web_sm")
        st.stop()
    
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    groq_client = Groq(api_key=GROQ_API_KEY)
    return nlp, embedder, groq_client

nlp, embedder, groq_client = load_models()

# ---------------------- CHROMADB SETUP ----------------------
@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB with proper error handling"""
    CHROMA_DB_PATH = "./chroma_db"
    
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name="constitution",
            metadata={"hnsw:space": "cosine"}
        )
        return client, collection
    except Exception as e:
        st.warning(f"ChromaDB initialization issue: {e}. Creating fresh database.")
        import shutil
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name="constitution",
            metadata={"hnsw:space": "cosine"}
        )
        return client, collection

chroma_client, collection = init_chromadb()

# ---------------------- PDF & KNOWLEDGE BASE ----------------------
def get_pdf_path():
    """Find PDF in multiple possible locations"""
    possible_paths = [
        "COK.pdf",
        "./COK.pdf",
        os.path.join(os.path.dirname(__file__), "COK.pdf"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    st.error("⚠️ COK.pdf not found. Please upload the Kenyan Constitution PDF.")
    st.stop()

PDF_PATH = get_pdf_path()

@st.cache_data
def extract_text_from_pdf(pdf_path):
    """Extract and cache PDF text"""
    with pdfplumber.open(pdf_path) as pdf:
        return "".join([page.extract_text() or "" for page in pdf.pages])

def chunk_text(text, max_tokens=500):
    """Split text into semantic chunks"""
    doc = nlp(text)
    chunks, current_chunk, current_length = [], "", 0
    
    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue
            
        sent_tokens = len(sent_text.split())  # Approximate token count
        
        if current_length + sent_tokens > max_tokens and current_chunk:
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
    """Embed chunks and store in ChromaDB"""
    with st.spinner("Embedding document chunks..."):
        for i in range(0, len(chunks), 50):  # Batch processing
            batch = chunks[i:i+50]
            embeddings = embedder.encode(batch)
            
            collection.add(
                documents=batch,
                embeddings=embeddings.tolist(),
                ids=[str(uuid.uuid4()) for _ in batch]
            )

def query_constitution(query, n_results=5):
    """Query the knowledge base"""
    query_embedding = embedder.encode([query])[0]
    results = collection.query(
        query_embeddings=[query_embedding.tolist()], 
        n_results=n_results
    )
    return results['documents'][0] if results['documents'] else []

def generate_response(query, context):
    """Generate response using Groq API"""
    context_text = "\n\n".join(context) if context else "No relevant context found."
    
    prompt = f"""You are a legal assistant specializing in the Kenyan Constitution.

Context from the Constitution:
{context_text}

User Question: {query}

Provide a clear, accurate answer based on the context above. If the context doesn't contain enough information, say so clearly."""

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a knowledgeable legal assistant specializing in Kenyan constitutional law. Provide accurate, helpful answers."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=800,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

@st.cache_resource
def setup_knowledge_base():
    """Initialize knowledge base on first run"""
    if collection.count() == 0:
        with st.spinner("🔄 Setting up knowledge base... This may take a minute."):
            text = extract_text_from_pdf(PDF_PATH)
            chunks = chunk_text(text)
            embed_and_store(chunks)
            st.success(f"✅ Knowledge base ready! Indexed {len(chunks)} chunks.")
    return True

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(
    page_title="Kenyan Constitution Chatbot", 
    page_icon="🇰🇪", 
    layout="centered"
)

st.markdown("""
<style>
body { 
    background-color: #121212; 
    color: white; 
}
.stTextArea textarea {
    background-color: #2a2a2a !important; 
    color: white !important;
    border: 2px solid #444 !important; 
    border-radius: 8px !important;
}
.response-box {
    background: linear-gradient(135deg, #1e1e1e 0%, #2a2a2a 100%);
    padding: 20px; 
    border-left: 6px solid #dc143c;
    border-radius: 8px; 
    min-height: 100px; 
    white-space: pre-wrap;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
}
.header {
    text-align: center;
    padding: 20px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header">
    <h1>🇰🇪 Kenyan Constitution Chatbot</h1>
    <p>Ask questions about Kenya's Constitution</p>
</div>
""", unsafe_allow_html=True)

# Initialize knowledge base
kb_ready = setup_knowledge_base()

# Language selection
language = st.selectbox("🌍 Language:", ["English", "Swahili"], index=0)

# Question input
question = st.text_area(
    "💬 Your Question:",
    placeholder="e.g., What are the fundamental rights in Kenya?",
    height=100
)

# Ask button
if st.button("🔍 Ask", use_container_width=True, type="primary"):
    if not question.strip():
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Thinking..."):
            try:
                # Query and generate response
                context = query_constitution(question)
                answer = generate_response(question, context)
                
                # Display with typing effect
                placeholder = st.empty()
                typed_text = ""
                
                for char in answer:
                    typed_text += char
                    placeholder.markdown(
                        f'<div class="response-box">{typed_text}▋</div>', 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.01)
                
                # Final output without cursor
                placeholder.markdown(
                    f'<div class="response-box">{answer}</div>', 
                    unsafe_allow_html=True
                )
                
                # Show sources
                with st.expander("📚 View Sources"):
                    for i, ctx in enumerate(context, 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(ctx[:300] + "..." if len(ctx) > 300 else ctx)
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 0.9em;'>
    <p>💡 Tip: Ask specific questions for better results</p>
    <p>Powered by Groq AI | Data: Constitution of Kenya 2010</p>
</div>
""", unsafe_allow_html=True)
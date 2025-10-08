# ---------------------- IMPORTS ----------------------
import os
import time
import warnings
import streamlit as st
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer
import chromadb
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
            
        sent_tokens = len(sent_text.split())
        
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
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i+50]
        embeddings = embedder.encode(batch)
        
        collection.add(
            documents=batch,
            embeddings=embeddings.tolist(),
            ids=[str(uuid.uuid4()) for _ in batch]
        )
        
        progress = min((i + 50) / len(chunks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {int(progress * 100)}% complete")
    
    progress_bar.empty()
    status_text.empty()

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
    page_title="Kenyan Constitution Assistant", 
    page_icon="🇰🇪", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Kenyan Flag Colors: Black (#000000), Red (#B91C1C), Green (#166534), White (#FFFFFF)
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header with Kenyan flag stripe */
    .main-header {
        background: linear-gradient(to right, 
            #000000 0%, #000000 25%,
            #B91C1C 25%, #B91C1C 50%,
            #166534 50%, #166534 75%,
            #FFFFFF 75%, #FFFFFF 100%
        );
        height: 8px;
        width: 100%;
        margin-bottom: 30px;
        border-radius: 4px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    
    /* Title section */
    .title-container {
        text-align: center;
        padding: 40px 20px 30px 20px;
        margin-bottom: 40px;
        background: linear-gradient(135deg, rgba(22, 101, 52, 0.1) 0%, rgba(185, 28, 28, 0.1) 100%);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .title-container h1 {
        color: #FFFFFF;
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        background: linear-gradient(135deg, #FFFFFF 0%, #E5E5E5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .title-container p {
        color: #B0B0B0;
        font-size: 1.2rem;
        margin-top: 10px;
        font-weight: 300;
    }
    
    .flag-emoji {
        font-size: 4rem;
        margin-bottom: 10px;
        filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3));
    }
    
    /* Input container */
    .input-section {
        background: rgba(26, 26, 26, 0.8);
        padding: 30px;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    
    /* Text area styling */
    .stTextArea textarea {
        background: rgba(42, 42, 42, 0.9) !important;
        color: #FFFFFF !important;
        border: 2px solid rgba(22, 101, 52, 0.5) !important;
        border-radius: 12px !important;
        font-size: 1.1rem !important;
        padding: 15px !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextArea textarea:focus {
        border-color: #166534 !important;
        box-shadow: 0 0 20px rgba(22, 101, 52, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #166534 0%, #22c55e 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 15px 40px !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(22, 101, 52, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(22, 101, 52, 0.6) !important;
        background: linear-gradient(135deg, #22c55e 0%, #166534 100%) !important;
    }
    
    /* Response box */
    .response-box {
        background: linear-gradient(145deg, rgba(26, 26, 26, 0.95) 0%, rgba(42, 42, 42, 0.95) 100%);
        padding: 30px;
        border-left: 6px solid #B91C1C;
        border-radius: 15px;
        min-height: 150px;
        white-space: pre-wrap;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #E5E5E5;
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
        margin: 20px 0;
        border: 1px solid rgba(185, 28, 28, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .response-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, 
            transparent 0%,
            #B91C1C 25%,
            #166534 75%,
            transparent 100%
        );
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 20px;
        margin: 30px 0;
        flex-wrap: wrap;
    }
    
    .stat-card {
        flex: 1;
        min-width: 200px;
        background: linear-gradient(135deg, rgba(22, 101, 52, 0.2) 0%, rgba(185, 28, 28, 0.2) 100%);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(22, 101, 52, 0.3);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #22c55e;
        margin: 10px 0;
    }
    
    .stat-label {
        color: #B0B0B0;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(22, 101, 52, 0.2) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(22, 101, 52, 0.3) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(26, 26, 26, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    /* Source cards */
    .source-card {
        background: rgba(42, 42, 42, 0.6);
        padding: 15px;
        border-radius: 10px;
        border-left: 3px solid #166534;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        background: rgba(42, 42, 42, 0.8);
        border-left-width: 5px;
    }
    
    /* Loading animation */
    .stSpinner > div {
        border-top-color: #166534 !important;
    }
    
    /* Warning and error boxes */
    .stAlert {
        border-radius: 12px !important;
        border-left: 5px solid #B91C1C !important;
    }
    
    /* Footer */
    .custom-footer {
        text-align: center;
        padding: 40px 20px;
        margin-top: 60px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        color: #888;
    }
    
    .custom-footer p {
        margin: 8px 0;
        font-size: 0.95rem;
    }
    
    .footer-logo {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #166534 0%, #22c55e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Smooth animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Header with Kenyan flag stripe
st.markdown('<div class="main-header"></div>', unsafe_allow_html=True)

# Title section
st.markdown("""
<div class="title-container animate-in">
    <div class="flag-emoji">🇰🇪</div>
    <h1>Kenyan Constitution Assistant</h1>
    <p>Your AI-powered legal companion for constitutional queries</p>
</div>
""", unsafe_allow_html=True)

# Initialize knowledge base
kb_ready = setup_knowledge_base()

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Question input
    st.markdown('<div class="input-section animate-in">', unsafe_allow_html=True)
    question = st.text_area(
        "💬 Ask Your Question",
        placeholder="e.g., What are the fundamental rights and freedoms in Kenya?",
        height=150,
        label_visibility="visible"
    )
    
    # Ask button
    ask_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Stats cards
    st.markdown("""
    <div class="stat-card animate-in" style="animation-delay: 0.2s;">
        <div class="stat-label">📚 Knowledge Base</div>
        <div class="stat-number">{}</div>
        <div class="stat-label">Document Chunks</div>
    </div>
    """.format(collection.count()), unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stat-card animate-in" style="animation-delay: 0.3s; margin-top: 20px;">
        <div class="stat-label">🤖 AI Model</div>
        <div style="color: #22c55e; font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Llama 3.3 70B</div>
        <div class="stat-label">Powered by Groq</div>
    </div>
    """, unsafe_allow_html=True)

# Process question
if ask_button:
    if not question.strip():
        st.warning("⚠️ Please enter a question to continue.")
    else:
        with st.spinner("🤔 Analyzing your question..."):
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
                        f'<div class="response-box animate-in">{typed_text}<span style="color: #22c55e;">▋</span></div>', 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.008)
                
                # Final output without cursor
                placeholder.markdown(
                    f'<div class="response-box animate-in">{answer}</div>', 
                    unsafe_allow_html=True
                )
                
                # Show sources in expandable section
                with st.expander("📚 View Supporting Sources", expanded=False):
                    if context:
                        for i, ctx in enumerate(context, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong style="color: #22c55e;">📄 Source {i}</strong><br/>
                                <span style="color: #E5E5E5; font-size: 0.95rem;">
                                {ctx[:400] + "..." if len(ctx) > 400 else ctx}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No specific sources found for this query.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please try rephrasing your question or contact support if the issue persists.")

# Footer
st.markdown("""
<div class="custom-footer">
    <div class="footer-logo">SYNERGY</div>
    <p>💡 <strong>Pro Tip:</strong> Ask specific questions for more accurate answers</p>
    <p style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
        Powered by <strong style="color: #22c55e;">Groq AI</strong> • 
        Data Source: <strong>Constitution of Kenya 2010</strong>
    </p>
    <p style="font-size: 0.85rem; color: #666;">
        Developed with 🇰🇪 by Synergy • © 2025 All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
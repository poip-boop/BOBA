# ---------------------- IMPORTS ----------------------
import os
import time
import warnings
import streamlit as st
import pdfplumber
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq
import uuid
from dotenv import load_dotenv
import re
from typing import List, Dict
import shutil
import sys

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
    st.info("""
    To set up your API key:
    1. Go to your Streamlit app dashboard
    2. Click on 'Settings' → 'Secrets'
    3. Add: `GROQ_API_KEY = "your-api-key-here"`
    4. Get your API key from https://console.groq.com
    """)
    st.stop()

# ---------------------- INITIALIZATION FLAGS ----------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.kb_ready = False

# ---------------------- CACHED RESOURCES ----------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """Load heavy models once and cache them"""
    try:
        # Try to load spacy model
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # If model not found, download it
            os.system("python -m spacy download en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        # Load embedding model - using a smaller model for faster loading
        with st.spinner("Loading embedding model..."):
            embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Smaller, faster model
        
        # Load reranking model
        with st.spinner("Loading reranking model..."):
            reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize Groq client
        groq_client = Groq(api_key=GROQ_API_KEY)
        
        return nlp, embedder, reranker, groq_client
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.stop()

# ---------------------- CHROMADB SETUP ----------------------
@st.cache_resource(show_spinner=False)
def init_chromadb():
    """Initialize ChromaDB with proper error handling"""
    # Use temp directory for Streamlit Cloud
    CHROMA_DB_PATH = "/tmp/chroma_db" if os.path.exists("/tmp") else "./chroma_db"
    
    try:
        # Clear any existing database to avoid conflicts
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
        
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        collection = client.get_or_create_collection(
            name="constitution",
            metadata={"hnsw:space": "cosine"}
        )
        return client, collection
    except Exception as e:
        st.error(f"ChromaDB initialization failed: {e}")
        # Fallback to in-memory database
        client = chromadb.Client()
        collection = client.get_or_create_collection(
            name="constitution",
            metadata={"hnsw:space": "cosine"}
        )
        return client, collection

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
    
    # If PDF not found, provide upload option
    st.error("⚠️ COK.pdf not found in the repository.")
    st.info("Please add COK.pdf to your GitHub repository root directory.")
    
    # Provide manual upload option
    uploaded_file = st.file_uploader("Or upload the Constitution PDF here:", type=['pdf'])
    if uploaded_file:
        # Save uploaded file
        with open("COK.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        return "COK.pdf"
    
    st.stop()

@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_path):
    """Extract and cache PDF text with better structure preservation"""
    text_chunks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    text_chunks.append({
                        'text': text,
                        'page': page_num + 1
                    })
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        st.stop()
    return text_chunks

def improved_chunk_text(text_data: List[Dict], chunk_size=400, overlap=100):
    """
    Improved chunking strategy for legal documents
    """
    chunks = []
    
    for page_data in text_data:
        text = page_data['text']
        page_num = page_data['page']
        
        # Split by articles/sections first (legal document structure)
        article_pattern = r'(Article \d+|ARTICLE \d+|Chapter \d+|CHAPTER \d+|Section \d+|SECTION \d+)'
        sections = re.split(article_pattern, text)
        
        current_chunk = ""
        current_metadata = {"page": page_num, "section": ""}
        
        for i, section in enumerate(sections):
            # Check if this is a section header
            if re.match(article_pattern, section):
                # Save previous chunk if it exists
                if current_chunk.strip():
                    chunks.append({
                        'text': current_chunk.strip(),
                        'metadata': current_metadata.copy()
                    })
                
                current_metadata['section'] = section
                current_chunk = section + " "
            else:
                # Add text to current chunk
                words = section.split()
                for word in words:
                    current_chunk += word + " "
                    
                    # Check if chunk is getting too large
                    if len(current_chunk.split()) > chunk_size:
                        # Save chunk with overlap
                        chunks.append({
                            'text': current_chunk.strip(),
                            'metadata': current_metadata.copy()
                        })
                        
                        # Create overlap
                        overlap_words = current_chunk.split()[-overlap:]
                        current_chunk = " ".join(overlap_words) + " "
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                'text': current_chunk.strip(),
                'metadata': current_metadata.copy()
            })
    
    return chunks

def embed_and_store(chunks: List[Dict], embedder, collection):
    """Embed chunks and store in ChromaDB with metadata"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_size = 10  # Smaller batch size for stability
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        texts = [chunk['text'] for chunk in batch]
        
        try:
            embeddings = embedder.encode(texts, show_progress_bar=False)
            
            # Prepare metadata
            metadatas = []
            for chunk in batch:
                metadata = {
                    'page': str(chunk['metadata']['page']),
                    'section': chunk['metadata'].get('section', '')
                }
                metadatas.append(metadata)
            
            collection.add(
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas,
                ids=[str(uuid.uuid4()) for _ in batch]
            )
        except Exception as e:
            st.warning(f"Batch processing error: {e}. Continuing...")
            continue
        
        progress = min((i + batch_size) / len(chunks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {int(progress * 100)}% complete")
    
    progress_bar.empty()
    status_text.empty()

def setup_knowledge_base(embedder, collection, pdf_path):
    """Initialize knowledge base on first run"""
    if collection.count() == 0:
        with st.spinner("🔄 Setting up knowledge base... This may take a minute."):
            text_data = extract_text_from_pdf(pdf_path)
            chunks = improved_chunk_text(text_data, chunk_size=400, overlap=100)
            embed_and_store(chunks, embedder, collection)
            st.success(f"✅ Knowledge base ready! Indexed {len(chunks)} chunks.")
            return True
    return True

# ---------------------- QUERY FUNCTIONS ----------------------
def generate_query_variations(query: str, groq_client) -> List[str]:
    """Generate multiple query variations for better retrieval"""
    variations = [query]
    
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Generate 2 alternative phrasings of the question that maintain the same meaning."},
                {"role": "user", "content": f"Original: {query}\n\nVariations:"}
            ],
            model="llama-3.1-8b-instant",
            max_tokens=150,
            temperature=0.7
        )
        
        new_queries = response.choices[0].message.content.strip().split('\n')
        variations.extend([q.strip('- 123.') for q in new_queries if q.strip()][:2])
    except:
        pass
    
    return variations[:3]

def hybrid_search(query: str, embedder, collection, groq_client, n_results=15) -> List[Dict]:
    """Hybrid search combining multiple query variations and semantic search"""
    all_results = []
    seen_docs = set()
    
    # Generate query variations
    query_variations = generate_query_variations(query, groq_client)
    
    # Search with each variation
    for q in query_variations:
        try:
            query_embedding = embedder.encode([q])[0]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            if results['documents'] and results['documents'][0]:
                for doc, metadata, distance in zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                ):
                    doc_hash = hash(doc[:100])
                    if doc_hash not in seen_docs:
                        seen_docs.add(doc_hash)
                        all_results.append({
                            'document': doc,
                            'metadata': metadata,
                            'distance': distance,
                            'query': q
                        })
        except Exception as e:
            st.warning(f"Search error: {e}")
            continue
    
    return all_results

def rerank_results(query: str, results: List[Dict], reranker, top_k=5) -> List[Dict]:
    """Rerank results using CrossEncoder for better relevance"""
    if not results:
        return []
    
    try:
        # Prepare pairs for reranking
        pairs = [[query, result['document']] for result in results]
        
        # Get reranking scores
        scores = reranker.predict(pairs)
        
        # Add scores to results
        for result, score in zip(results, scores):
            result['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(results, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked[:top_k]
    except Exception as e:
        st.warning(f"Reranking error: {e}. Using original results.")
        return results[:top_k]

def query_constitution(query: str, embedder, collection, reranker, groq_client, n_results=5) -> List[Dict]:
    """Main retrieval function with multi-step querying"""
    # Step 1: Hybrid search
    initial_results = hybrid_search(query, embedder, collection, groq_client, n_results=15)
    
    # Step 2: Rerank results
    reranked_results = rerank_results(query, initial_results, reranker, top_k=n_results)
    
    return reranked_results

def generate_response(query: str, context: List[Dict], groq_client) -> str:
    """Generate response using the best LLM model with improved prompting"""
    if not context:
        return "I couldn't find relevant information in the Constitution to answer your question. Please try rephrasing or ask about a different topic."
    
    # Format context with metadata
    context_text = ""
    for i, ctx in enumerate(context, 1):
        section = ctx['metadata'].get('section', 'Unknown')
        page = ctx['metadata'].get('page', 'Unknown')
        context_text += f"\n[Source {i} - {section}, Page {page}]\n{ctx['document']}\n"
    
    prompt = f"""You are a legal expert specializing in Kenyan Constitutional Law. Answer the user's question using ONLY the provided context from the Constitution of Kenya 2010.

CONTEXT FROM THE CONSTITUTION:
{context_text}

USER QUESTION: {query}

INSTRUCTIONS:
1. Provide a clear, accurate answer based STRICTLY on the context above
2. Cite specific Articles, Chapters, or Sections when possible
3. If the context doesn't fully answer the question, state what information is available and what is missing
4. Use clear, accessible language while maintaining legal accuracy
5. Structure your answer with paragraphs for readability
6. Do NOT make up information not present in the context

ANSWER:"""

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert legal assistant specializing in Kenyan Constitutional Law."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=1500,
            temperature=0.1,
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        # Fallback to smaller model if main model fails
        try:
            response = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are an expert legal assistant."},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.1-8b-instant",
                max_tokens=1000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except:
            return f"Error generating response: {str(e)}\n\nPlease try again or rephrase your question."

# ---------------------- STREAMLIT UI ----------------------
st.set_page_config(
    page_title="Legal Assistant", 
    page_icon="🇰🇪", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS styling (shortened for deployment)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); }
    .main-header { background: linear-gradient(to right, #000000 0%, #000000 25%, #B91C1C 25%, #B91C1C 50%, #166534 50%, #166534 75%, #FFFFFF 75%, #FFFFFF 100%); height: 8px; width: 100%; margin-bottom: 30px; border-radius: 4px; }
    .title-container { text-align: center; padding: 40px 20px 30px 20px; margin-bottom: 40px; background: linear-gradient(135deg, rgba(22, 101, 52, 0.1) 0%, rgba(185, 28, 28, 0.1) 100%); border-radius: 20px; }
    .title-container h1 { color: #FFFFFF; font-size: 3rem; font-weight: 700; margin: 0; }
    .title-container p { color: #B0B0B0; font-size: 1.2rem; margin-top: 10px; }
    .response-box { background: linear-gradient(145deg, rgba(26, 26, 26, 0.95) 0%, rgba(42, 42, 42, 0.95) 100%); padding: 30px; border-left: 6px solid #B91C1C; border-radius: 15px; color: #E5E5E5; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="title-container">
    <h1>🇰🇪 Kenyan Legal Assistant</h1>
    <p>AI-powered legal companion</p>
</div>
""", unsafe_allow_html=True)

# Initialize models and database
if not st.session_state.initialized:
    with st.spinner("🚀 Initializing AI models... This may take a moment on first run."):
        try:
            nlp, embedder, reranker, groq_client = load_models()
            chroma_client, collection = init_chromadb()
            PDF_PATH = get_pdf_path()
            st.session_state.initialized = True
            st.session_state.nlp = nlp
            st.session_state.embedder = embedder
            st.session_state.reranker = reranker
            st.session_state.groq_client = groq_client
            st.session_state.collection = collection
            st.session_state.pdf_path = PDF_PATH
        except Exception as e:
            st.error(f"Initialization failed: {str(e)}")
            st.stop()

# Setup knowledge base
if st.session_state.initialized and not st.session_state.kb_ready:
    st.session_state.kb_ready = setup_knowledge_base(
        st.session_state.embedder, 
        st.session_state.collection, 
        st.session_state.pdf_path
    )

# Main interface
col1, col2 = st.columns([3, 1])

with col1:
    question = st.text_area(
        "💬 Ask Your Question",
        placeholder="e.g., What are the fundamental rights and freedoms guaranteed in the Constitution?",
        height=120
    )
    
    ask_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")

with col2:
    st.metric("📚 Knowledge Base", f"{st.session_state.collection.count()} chunks")
    st.metric("🤖 AI Model", "Llama 3.3 70B")

if ask_button and question:
    with st.spinner("🤔 Analyzing your question..."):
        try:
            context_results = query_constitution(
                question,
                st.session_state.embedder,
                st.session_state.collection,
                st.session_state.reranker,
                st.session_state.groq_client
            )
            answer = generate_response(
                question, 
                context_results,
                st.session_state.groq_client
            )
            
            st.markdown(f'<div class="response-box">{answer}</div>', unsafe_allow_html=True)
            
            with st.expander("📚 View Supporting Sources", expanded=False):
                if context_results:
                    for i, ctx in enumerate(context_results, 1):
                        st.write(f"**Source {i}** - {ctx['metadata'].get('section', 'General')} (Page {ctx['metadata'].get('page', 'N/A')})")
                        st.write(ctx['document'][:500] + "..." if len(ctx['document']) > 500 else ctx['document'])
                        st.divider()
                        
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")
            st.info("Please try rephrasing your question.")

elif ask_button:
    st.warning("⚠️ Please enter a question to continue.")

# Footer
st.markdown("---")
st.markdown("Powered by Groq AI • Constitution of Kenya 2010 • Developed by Synergy", unsafe_allow_html=True)
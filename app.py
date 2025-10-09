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
    
    # Better embedding model for legal text
    embedder = SentenceTransformer('BAAI/bge-small-en-v1.5')
    
    # Free reranking model for better relevance
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    groq_client = Groq(api_key=GROQ_API_KEY)
    return nlp, embedder, reranker, groq_client

nlp, embedder, reranker, groq_client = load_models()

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

# ---------------------- QUERY TRANSFORMATION ----------------------
def generate_query_variations(query: str) -> List[str]:
    """Generate multiple query variations for better retrieval"""
    variations = [query]
    
    # Use LLM to generate query variations
    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a query expansion expert. Generate 2 alternative phrasings of the user's question that maintain the same meaning but use different legal terminology."},
                {"role": "user", "content": f"Original question: {query}\n\nGenerate 2 variations (one per line):"}
            ],
            model="llama-3.1-8b-instant",  # Faster model for query transformation
            max_tokens=150,
            temperature=0.7
        )
        
        new_queries = response.choices[0].message.content.strip().split('\n')
        variations.extend([q.strip('- 123.') for q in new_queries if q.strip()])
    except:
        pass
    
    return variations[:3]  # Limit to 3 variations

def extract_legal_entities(query: str) -> List[str]:
    """Extract legal entities and keywords from query"""
    doc = nlp(query)
    entities = []
    
    # Extract named entities
    for ent in doc.ents:
        if ent.label_ in ['LAW', 'ORG', 'GPE', 'PERSON']:
            entities.append(ent.text)
    
    # Extract legal keywords
    legal_keywords = ['right', 'freedom', 'duty', 'article', 'chapter', 'constitution', 
                     'law', 'court', 'parliament', 'president', 'citizen', 'government']
    
    for token in doc:
        if token.text.lower() in legal_keywords:
            entities.append(token.text)
    
    return list(set(entities))

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
    """Extract and cache PDF text with better structure preservation"""
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            if text.strip():
                text_chunks.append({
                    'text': text,
                    'page': page_num + 1
                })
    return text_chunks

def improved_chunk_text(text_data: List[Dict], chunk_size=400, overlap=100):
    """
    Improved chunking strategy for legal documents:
    - Preserves article/section boundaries
    - Maintains context with overlap
    - Includes metadata (page numbers, sections)
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

def embed_and_store(chunks: List[Dict]):
    """Embed chunks and store in ChromaDB with metadata"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, len(chunks), 50):
        batch = chunks[i:i+50]
        texts = [chunk['text'] for chunk in batch]
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
        
        progress = min((i + 50) / len(chunks), 1.0)
        progress_bar.progress(progress)
        status_text.text(f"Processing: {int(progress * 100)}% complete")
    
    progress_bar.empty()
    status_text.empty()

def hybrid_search(query: str, n_results=15) -> List[Dict]:
    """
    Hybrid search combining:
    1. Multiple query variations
    2. Semantic search
    3. Keyword boosting
    """
    all_results = []
    seen_docs = set()
    
    # Generate query variations
    query_variations = generate_query_variations(query)
    
    # Search with each variation
    for q in query_variations:
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
                doc_hash = hash(doc[:100])  # Use first 100 chars as hash
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'distance': distance,
                        'query': q
                    })
    
    return all_results

def rerank_results(query: str, results: List[Dict], top_k=5) -> List[Dict]:
    """
    Rerank results using CrossEncoder for better relevance
    """
    if not results:
        return []
    
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

def query_constitution(query: str, n_results=5) -> List[Dict]:
    """
    Main retrieval function with multi-step querying:
    1. Query transformation
    2. Hybrid search
    3. Reranking
    """
    # Step 1: Hybrid search
    initial_results = hybrid_search(query, n_results=15)
    
    # Step 2: Rerank results
    reranked_results = rerank_results(query, initial_results, top_k=n_results)
    
    return reranked_results

def generate_response(query: str, context: List[Dict]) -> str:
    """
    Generate response using the best LLM model with improved prompting
    """
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
                {"role": "system", "content": "You are an expert legal assistant specializing in Kenyan Constitutional Law. You provide accurate, well-structured answers based on the Constitution of Kenya 2010. Always cite specific articles and maintain professional legal language."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",  # Best model for legal reasoning
            max_tokens=1500,
            temperature=0.1,  # Low temperature for factual accuracy
            top_p=0.9
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}\n\nPlease try again or rephrase your question."

@st.cache_resource
def setup_knowledge_base():
    """Initialize knowledge base on first run"""
    if collection.count() == 0:
        with st.spinner("🔄 Setting up knowledge base... This may take a minute."):
            text_data = extract_text_from_pdf(PDF_PATH)
            chunks = improved_chunk_text(text_data, chunk_size=400, overlap=100)
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

# [Previous CSS styling remains the same]
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .main-header { background: linear-gradient(to right, #000000 0%, #000000 25%, #B91C1C 25%, #B91C1C 50%, #166534 50%, #166534 75%, #FFFFFF 75%, #FFFFFF 100%); height: 8px; width: 100%; margin-bottom: 30px; border-radius: 4px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
    .title-container { text-align: center; padding: 40px 20px 30px 20px; margin-bottom: 40px; background: linear-gradient(135deg, rgba(22, 101, 52, 0.1) 0%, rgba(185, 28, 28, 0.1) 100%); border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); }
    .title-container h1 { color: #FFFFFF; font-size: 3rem; font-weight: 700; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.5); background: linear-gradient(135deg, #FFFFFF 0%, #E5E5E5 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .title-container p { color: #B0B0B0; font-size: 1.2rem; margin-top: 10px; font-weight: 300; }
    .flag-emoji { font-size: 4rem; margin-bottom: 10px; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3)); }
    .input-section { background: rgba(26, 26, 26, 0.8); padding: 30px; border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1); margin-bottom: 30px; backdrop-filter: blur(10px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
    .stTextArea textarea { background: rgba(42, 42, 42, 0.9) !important; color: #FFFFFF !important; border: 2px solid rgba(22, 101, 52, 0.5) !important; border-radius: 12px !important; font-size: 1.1rem !important; padding: 15px !important; transition: all 0.3s ease !important; }
    .stTextArea textarea:focus { border-color: #166534 !important; box-shadow: 0 0 20px rgba(22, 101, 52, 0.3) !important; }
    .stButton > button { background: linear-gradient(135deg, #166534 0%, #22c55e 100%) !important; color: white !important; border: none !important; border-radius: 12px !important; padding: 15px 40px !important; font-size: 1.1rem !important; font-weight: 600 !important; transition: all 0.3s ease !important; box-shadow: 0 4px 15px rgba(22, 101, 52, 0.4) !important; text-transform: uppercase !important; letter-spacing: 1px !important; }
    .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 25px rgba(22, 101, 52, 0.6) !important; background: linear-gradient(135deg, #22c55e 0%, #166534 100%) !important; }
    .response-box { background: linear-gradient(145deg, rgba(26, 26, 26, 0.95) 0%, rgba(42, 42, 42, 0.95) 100%); padding: 30px; border-left: 6px solid #B91C1C; border-radius: 15px; min-height: 150px; white-space: pre-wrap; font-size: 1.1rem; line-height: 1.8; color: #E5E5E5; box-shadow: 0 8px 32px rgba(0,0,0,0.4); margin: 20px 0; border: 1px solid rgba(185, 28, 28, 0.3); position: relative; overflow: hidden; }
    .stat-card { flex: 1; min-width: 200px; background: linear-gradient(135deg, rgba(22, 101, 52, 0.2) 0%, rgba(185, 28, 28, 0.2) 100%); padding: 20px; border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); text-align: center; transition: all 0.3s ease; }
    .stat-card:hover { transform: translateY(-5px); box-shadow: 0 10px 30px rgba(22, 101, 52, 0.3); }
    .stat-number { font-size: 2.5rem; font-weight: 700; color: #22c55e; margin: 10px 0; }
    .stat-label { color: #B0B0B0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; }
    .streamlit-expanderHeader { background: rgba(22, 101, 52, 0.2) !important; border-radius: 10px !important; border: 1px solid rgba(22, 101, 52, 0.3) !important; color: #FFFFFF !important; font-weight: 600 !important; }
    .source-card { background: rgba(42, 42, 42, 0.6); padding: 15px; border-radius: 10px; border-left: 3px solid #166534; margin: 10px 0; transition: all 0.3s ease; }
    .custom-footer { text-align: center; padding: 40px 20px; margin-top: 60px; border-top: 1px solid rgba(255, 255, 255, 0.1); color: #888; }
    .footer-logo { font-size: 1.5rem; font-weight: 700; background: linear-gradient(135deg, #166534 0%, #22c55e 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .animate-in { animation: fadeIn 0.6s ease-out; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"></div>', unsafe_allow_html=True)

st.markdown("""
<div class="title-container animate-in">
    <div class="flag-emoji">🇰🇪</div>
    <h1>Kenyan Constitution Assistant</h1>
    <p>Advanced AI-powered legal companion with enhanced RAG pipeline</p>
</div>
""", unsafe_allow_html=True)

kb_ready = setup_knowledge_base()

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="input-section animate-in">', unsafe_allow_html=True)
    question = st.text_area(
        "💬 Ask Your Question",
        placeholder="e.g., What are the fundamental rights and freedoms guaranteed in the Constitution?",
        height=150,
        label_visibility="visible"
    )
    
    ask_button = st.button("🔍 Get Answer", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="stat-card animate-in" style="animation-delay: 0.2s;">
        <div class="stat-label">📚 Knowledge Base</div>
        <div class="stat-number">{collection.count()}</div>
        <div class="stat-label">Document Chunks</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="stat-card animate-in" style="animation-delay: 0.3s; margin-top: 20px;">
        <div class="stat-label">🤖 AI Model</div>
        <div style="color: #22c55e; font-size: 1.2rem; font-weight: 600; margin: 10px 0;">Llama 3.3 70B</div>
        <div class="stat-label">With Reranking</div>
    </div>
    """, unsafe_allow_html=True)

if ask_button:
    if not question.strip():
        st.warning("⚠️ Please enter a question to continue.")
    else:
        with st.spinner("🤔 Analyzing your question with multi-step retrieval..."):
            try:
                context_results = query_constitution(question)
                answer = generate_response(question, context_results)
                
                placeholder = st.empty()
                typed_text = ""
                
                for char in answer:
                    typed_text += char
                    placeholder.markdown(
                        f'<div class="response-box animate-in">{typed_text}<span style="color: #22c55e;">▋</span></div>', 
                        unsafe_allow_html=True
                    )
                    time.sleep(0.008)
                
                placeholder.markdown(
                    f'<div class="response-box animate-in">{answer}</div>', 
                    unsafe_allow_html=True
                )
                
                with st.expander("📚 View Supporting Sources with Relevance Scores", expanded=False):
                    if context_results:
                        for i, ctx in enumerate(context_results, 1):
                            section = ctx['metadata'].get('section', 'General')
                            page = ctx['metadata'].get('page', 'N/A')
                            score = ctx.get('rerank_score', 0)
                            
                            st.markdown(f"""
                            <div class="source-card">
                                <strong style="color: #22c55e;">📄 Source {i}</strong>
                                <span style="color: #888; font-size: 0.85rem;"> | {section} | Page {page} | Relevance: {score:.3f}</span><br/>
                                <span style="color: #E5E5E5; font-size: 0.95rem;">
                                {ctx['document'][:500] + "..." if len(ctx['document']) > 500 else ctx['document']}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No specific sources found for this query.")
                        
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.info("Please try rephrasing your question or contact support if the issue persists.")

st.markdown("""
<div class="custom-footer">
    <div class="footer-logo">SYNERGY</div>
    <p>💡 <strong>Enhanced with:</strong> Query Transformation • Hybrid Search • Cross-Encoder Reranking</p>
    <p style="margin-top: 20px; padding-top: 20px; border-top: 1px solid rgba(255,255,255,0.1);">
        Powered by <strong style="color: #22c55e;">Groq AI (Llama 3.3 70B)</strong> • 
        Data: <strong>Constitution of Kenya 2010</strong>
    </p>
    <p style="font-size: 0.85rem; color: #666;">
        Developed with 🇰🇪 by Synergy • © 2025 All Rights Reserved
    </p>
</div>
""", unsafe_allow_html=True)
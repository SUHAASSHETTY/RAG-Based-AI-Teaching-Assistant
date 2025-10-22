# RAG based AI assistatt - Professional Theme

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import json
import os
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Video Content Q&A System",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS with Black and White/Red Theme
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: #000000;
        color: #ffffff;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: 2px;
        border-bottom: 3px solid #dc143c;
        padding-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #dc143c;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Cards */
    .metric-card {
        background: #1a1a1a;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #333333;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border-color: #dc143c;
        box-shadow: 0 6px 20px rgba(220, 20, 60, 0.3);
    }
    
    /* Result Cards */
    .result-card {
        background: #1a1a1a;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 2px solid #333333;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.5);
        color: #ffffff;
    }
    
    .result-card:hover {
        border-color: #dc143c;
    }
    
    /* Timestamp Badges */
    .timestamp-badge {
        background: #dc143c;
        color: #ffffff;
        padding: 0.5rem 1rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
        transition: all 0.3s ease;
    }
    
    .timestamp-badge:hover {
        background: #ff1744;
    }
    
    /* Similarity Badges */
    .similarity-badge-high {
        background: #ffffff;
        color: #000000;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 2px solid #ffffff;
    }
    
    .similarity-badge-medium {
        background: #dc143c;
        color: #ffffff;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 2px solid #dc143c;
    }
    
    .similarity-badge-low {
        background: transparent;
        color: #ffffff;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 2px solid #ffffff;
    }
    
    /* Status Messages */
    .status-success {
        background: #1a1a1a;
        border-left: 4px solid #ffffff;
        padding: 1rem;
        border-radius: 0.3rem;
        color: #ffffff;
        margin: 1rem 0;
    }
    
    .status-error {
        background: #1a1a1a;
        border-left: 4px solid #dc143c;
        padding: 1rem;
        border-radius: 0.3rem;
        color: #dc143c;
        margin: 1rem 0;
    }
    
    /* Custom Buttons */
    .stButton>button {
        background: #dc143c;
        color: #ffffff;
        border: none;
        border-radius: 0.3rem;
        padding: 0.7rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton>button:hover {
        background: #ff1744;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(220, 20, 60, 0.4);
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: #0a0a0a;
        border-right: 2px solid #333333;
    }
    
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #dc143c;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.9rem;
    }
    
    /* Text Input Areas */
    .stTextArea textarea {
        background: #1a1a1a;
        color: #ffffff;
        border: 2px solid #333333;
        border-radius: 0.3rem;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #dc143c;
        box-shadow: 0 0 10px rgba(220, 20, 60, 0.2);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #1a1a1a;
        border: 1px solid #333333;
        border-radius: 0.3rem;
        color: #ffffff;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #dc143c;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #dc143c;
        font-size: 2rem;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #999999;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: #dc143c;
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #1a1a1a;
        border: 2px solid #333333;
        border-radius: 0.3rem;
        color: #ffffff;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #dc143c;
        border-color: #dc143c;
        color: #ffffff;
    }
    
    /* Section Headers */
    h3 {
        color: #dc143c;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        border-bottom: 2px solid #333333;
        padding-bottom: 0.5rem;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #999999;
        padding: 2rem;
        border-top: 2px solid #333333;
        margin-top: 3rem;
        background: #0a0a0a;
        border-radius: 0.5rem;
    }
    
    .footer h4 {
        color: #dc143c;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #1a1a1a;
        border: 2px solid #333333;
        color: #ffffff;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: #dc143c;
    }
    
    /* Multiselect */
    .stMultiSelect > div > div {
        background: #1a1a1a;
        border: 2px solid #333333;
    }
    
    /* Multiselect selected items and text */
    .stMultiSelect [data-baseweb="tag"] {
        background: #dc143c;
        color: #ffffff;
    }
    
    .stMultiSelect input {
        color: #dc143c !important;
    }
    
    .stMultiSelect [role="button"] {
        color: #dc143c;
    }
    
    /* Info/Warning boxes */
    .stAlert {
        background: #1a1a1a;
        border: 1px solid #333333;
        color: #ffffff;
    }
    
    /* General text color */
    p, span, div {
        color: #ffffff;
    }
    
    /* Links */
    a {
        color: #dc143c;
    }
    
    a:hover {
        color: #ff1744;
    }
    
    /* Download button text fix */
    .stDownloadButton button {
        background: #dc143c !important;
        color: #ffffff !important;
    }
    
    .stDownloadButton button:hover {
        background: #ff1744 !important;
    }
    
    .stDownloadButton button p,
    .stDownloadButton button span,
    .stDownloadButton button div {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0

@st.cache_data
def load_embeddings():
    """Load the pre-computed embeddings"""
    try:
        if os.path.exists('embeddings.joblib'):
            df = joblib.load('embeddings.joblib')
            return df
        else:
            st.error("embeddings.joblib file not found. Please run preprocess_json.py first.")
            return None
    except Exception as e:
        st.error(f"Error loading embeddings: {str(e)}")
        return None

def create_embedding(text_list):
    """Create embeddings using Ollama API"""
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        }, timeout=30)
        embedding = r.json()["embeddings"] 
        return embedding
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        st.info("Make sure Ollama is running: ollama serve")
        return None

def generate_response(prompt, model="llama3.2", temperature=0.7):
    """Generate response using Ollama API"""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }, timeout=60)
        response = r.json()
        return response["response"]
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        st.info(f"Make sure Ollama is running and {model} model is installed")
        return None

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def search_similar_content(query, df, top_k=5):
    """Search for similar content based on query embedding"""
    if df is None or df.empty:
        return None
    
    query_embedding = create_embedding([query])
    if query_embedding is None:
        return None
    
    query_embedding = query_embedding[0]
    
    embeddings_matrix = np.vstack(df['embedding'].values)
    similarities = cosine_similarity(embeddings_matrix, [query_embedding]).flatten()
    
    top_indices = similarities.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    return results

def check_ollama_connection():
    """Check if Ollama is running and return available models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return True, [model['name'] for model in models]
        return False, []
    except:
        return False, []

def export_to_markdown(query, ai_response, results):
    """Export results to markdown format"""
    md_content = f"""# Video Q&A Search Results
    
**Query:** {query}
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## AI Response

{ai_response}

---

## Relevant Content Chunks

"""
    for idx, row in results.iterrows():
        start_time = format_timestamp(row.get('start', 0))
        end_time = format_timestamp(row.get('end', 0))
        md_content += f"""
### {row.get('title', 'Unknown')} - Video {row.get('number', 'N/A')}
- **Timestamp:** {start_time} - {end_time}
- **Similarity:** {row['similarity']:.3f}
- **Content:** {row.get('text', 'No text available')}

---
"""
    return md_content

def get_similarity_badge_class(similarity):
    """Return appropriate badge class based on similarity score"""
    if similarity > 0.7:
        return "similarity-badge-high"
    elif similarity > 0.5:
        return "similarity-badge-medium"
    else:
        return "similarity-badge-low"

def main():
    # Header
    st.markdown('<h1 class="main-header">RAG based AI teaching</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Intelligent Video Library Search</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### SYSTEM SETTINGS")
        
        # Connection Status
        is_connected, available_models = check_ollama_connection()
        
        if is_connected:
            st.markdown('<div class="status-success">● Ollama Connected</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error">● Ollama Disconnected</div>', unsafe_allow_html=True)
            st.info("Run: `ollama serve` in terminal")
        
        st.markdown("---")
        
        # Model Configuration
        st.markdown("#### Model Configuration")
        
        if available_models:
            default_models = ["llama3.2", "deepseek-r1"]
            model_list = list(set(default_models + available_models))
        else:
            model_list = ["llama3.2", "deepseek-r1"]
        
        selected_model = st.selectbox("AI Model", model_list, index=0)
        
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1,
                               help="Higher values make output more creative, lower values more focused")
        
        st.markdown("---")
        
        # Search Configuration
        st.markdown("#### Search Configuration")
        top_k = st.slider("Results to Show", min_value=1, max_value=15, value=5)
        
        min_similarity = st.slider("Min Similarity Threshold", min_value=0.0, max_value=1.0, value=0.0, step=0.05,
                                   help="Filter results below this similarity score")
        
        st.markdown("---")
        
        # Statistics
        st.markdown("#### Session Statistics")
        st.metric("Total Queries", st.session_state.query_count)
        
        if st.button("Clear History"):
            st.session_state.search_history = []
            st.session_state.query_count = 0
            st.rerun()
        
        st.markdown("---")
        
        # Search History
        if st.session_state.search_history:
            st.markdown("#### Recent Searches")
            for i, hist in enumerate(reversed(st.session_state.search_history[-5:])):
                with st.expander(f"Query {len(st.session_state.search_history) - i}"):
                    st.text(hist['query'][:100] + "..." if len(hist['query']) > 100 else hist['query'])
                    st.caption(f"Time: {hist['timestamp']}")
    
    # Load embeddings
    with st.spinner("Loading embeddings database..."):
        df = load_embeddings()
    
    if df is None:
        st.stop()
    
    # Dashboard Metrics
    st.markdown("### DATABASE OVERVIEW")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Chunks", f"{len(df):,}")
    
    with col2:
        unique_videos = df['title'].nunique() if 'title' in df.columns else 0
        st.metric("Unique Videos", unique_videos)
    
    with col3:
        total_duration = df['end'].sum() / 60 if 'end' in df.columns else 0
        st.metric("Total Duration", f"{total_duration:.0f} sec")
    
    with col4:
        avg_chunk_length = df['text'].str.len().mean() if 'text' in df.columns else 0
        st.metric("Avg Chunk Length", f"{avg_chunk_length:.0f} chars")
    
    st.markdown("---")
    
    # Main Query Interface
    st.markdown("### SEARCH INTERFACE")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["Text Query", "Advanced Search"])
    
    with tab1:
        query = st.text_area(
            "Enter your question:",
            placeholder="e.g., How to implement authentication in web applications?",
            height=120,
            key="main_query"
        )
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            search_clicked = st.button("Search", type="primary", use_container_width=True)
        with col2:
            example_query = st.button("Load Example", use_container_width=True)
        with col3:
            clear_query = st.button("Clear", use_container_width=True)
        
        if example_query:
            st.session_state.example_loaded = True
            st.rerun()
        
        if clear_query:
            st.session_state.main_query = ""
            st.rerun()
    
    with tab2:
        st.markdown("#### Filter by Video Properties")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_video = st.multiselect(
                "Select Videos",
                options=df['title'].unique().tolist() if 'title' in df.columns else [],
                default=None
            )
        
        with col2:
            filter_number = st.multiselect(
                "Select Video Numbers",
                options=sorted(df['number'].unique().tolist()) if 'number' in df.columns else [],
                default=None
            )
        
        advanced_query = st.text_area(
            "Enter your question:",
            placeholder="Search within selected filters...",
            height=100,
            key="advanced_query"
        )
        
        advanced_search = st.button("Search with Filters", type="primary")
    
    # Handle search
    if (search_clicked and query.strip()) or (advanced_search and advanced_query.strip()):
        current_query = query.strip() if search_clicked else advanced_query.strip()
        
        # Apply filters if in advanced mode
        filtered_df = df.copy()
        if advanced_search:
            if filter_video:
                filtered_df = filtered_df[filtered_df['title'].isin(filter_video)]
            if filter_number:
                filtered_df = filtered_df[filtered_df['number'].isin(filter_number)]
        
        # Progress indicator
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Creating query embedding...")
        progress_bar.progress(25)
        
        with st.spinner("Searching for relevant content..."):
            results = search_similar_content(current_query, filtered_df, top_k)
            progress_bar.progress(50)
            
            if results is not None and not results.empty:
                # Filter by minimum similarity
                results = results[results['similarity'] >= min_similarity]
                
                if results.empty:
                    st.warning(f"No results found with similarity >= {min_similarity}")
                    progress_bar.empty()
                    status_text.empty()
                else:
                    status_text.text("Generating AI response...")
                    progress_bar.progress(75)
                    
                    # Update statistics
                    st.session_state.query_count += 1
                    st.session_state.search_history.append({
                        'query': current_query,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'results_count': len(results)
                    })
                    
                    # Prepare context
                    context_data = results[["title", "number", "start", "end", "text"]].to_json(orient="records")
                    
                    prompt = f'''I am teaching Web Development in my course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, and the text at that time:

{context_data}
---------------------------------
User Question: "{current_query}"

You are a helpful teaching assistant. Based on the video chunks above, answer the user's question in a natural, conversational way. Guide them to the specific video(s) and timestamp(s) where they can find the relevant content. If the question is unrelated to the course content, politely inform them that you can only answer questions about the course material.'''

                    ai_response = generate_response(prompt, selected_model, temperature)
                    progress_bar.progress(100)
                    status_text.text("Complete")
                    time.sleep(0.5)
                    progress_bar.empty()
                    status_text.empty()
                    
                    if ai_response:
                        # Display results
                        st.markdown("---")
                        st.markdown("### AI RESPONSE")
                        
                        # AI Response Card
                        st.markdown(f"""
                        <div class="result-card">
                            {ai_response}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics
                        st.markdown("### SEARCH METRICS")
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            avg_sim = results['similarity'].mean()
                            st.metric("Avg Similarity", f"{avg_sim:.3f}")
                        
                        with metric_col2:
                            max_sim = results['similarity'].max()
                            st.metric("Max Similarity", f"{max_sim:.3f}")
                        
                        with metric_col3:
                            st.metric("Results Found", len(results))
                        
                        # Results
                        st.markdown("### RELEVANT CONTENT")
                        
                        for idx, row in results.iterrows():
                            badge_class = get_similarity_badge_class(row['similarity'])
                            
                            with st.expander(
                                f"{row.get('title', 'Unknown')} - Video {row.get('number', 'N/A')} | Similarity: {row['similarity']:.3f}",
                                expanded=(idx == results.index[0])
                            ):
                                col1, col2 = st.columns([3, 1])
                                
                                with col1:
                                    st.markdown(f"**Content:**")
                                    st.write(row.get('text', 'No text available'))
                                
                                with col2:
                                    start_time = format_timestamp(row.get('start', 0))
                                    end_time = format_timestamp(row.get('end', 0))
                                    
                                    st.markdown(f"""
                                    <div class="timestamp-badge">
                                        {start_time} - {end_time}
                                    </div>
                                    <br><br>
                                    <div class="{badge_class}">
                                        Score: {row['similarity']:.3f}
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        # Export Options
                        st.markdown("---")
                        st.markdown("### EXPORT RESULTS")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.download_button(
                                label="Download AI Response (TXT)",
                                data=ai_response,
                                file_name=f"ai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col2:
                            csv_data = results[['title', 'number', 'start', 'end', 'text', 'similarity']].to_csv(index=False)
                            st.download_button(
                                label="Download Results (CSV)",
                                data=csv_data,
                                file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        
                        with col3:
                            md_data = export_to_markdown(current_query, ai_response, results)
                            st.download_button(
                                label="Download Report (MD)",
                                data=md_data,
                                file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                                mime="text/markdown",
                                use_container_width=True
                            )
            else:
                progress_bar.empty()
                status_text.empty()
                st.error("No results found. Please check your Ollama connection.")
    
    elif (search_clicked or advanced_search):
        st.warning("Please enter a question before searching.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4>VIDEO CONTENT Q&A SYSTEM</h4>
        <p>Powered by Ollama | Built with Streamlit</p>
        <p style="font-size: 0.8rem;">© 2025 All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
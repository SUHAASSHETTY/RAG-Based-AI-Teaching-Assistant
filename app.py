# This is a frontend Streamlit app for a Video Content Q&A system

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import json
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Video Content Q&A System",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .video-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .timestamp {
        background-color: #dbeafe;
        padding: 0.2rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        color: #1e40af;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_embeddings():
    """Load the pre-computed embeddings"""
    try:
        if os.path.exists('embeddings.joblib'):
            df = joblib.load('embeddings.joblib')
            return df
        else:
            st.error("❌ embeddings.joblib file not found. Please run preprocess_json.py first.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading embeddings: {str(e)}")
        return None

def create_embedding(text_list):
    """Create embeddings using Ollama API - simplified for local use"""
    try:
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })
        embedding = r.json()["embeddings"] 
        return embedding
    except Exception as e:
        st.error(f"❌ Error creating embeddings: {str(e)}")
        st.info("💡 Make sure Ollama is running: ollama serve")
        return None

def generate_response(prompt, model="llama3.2"):
    """Generate response using Ollama API - simplified for local use"""
    try:
        r = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False
        })
        response = r.json()
        return response["response"]
    except Exception as e:
        st.error(f"❌ Error generating response: {str(e)}")
        st.info(f"💡 Make sure Ollama is running and {model} model is installed")
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
    
    # Create embedding for the query
    query_embedding = create_embedding([query])
    if query_embedding is None:
        return None
    
    query_embedding = query_embedding[0]
    
    # Calculate similarities
    embeddings_matrix = np.vstack(df['embedding'].values)
    similarities = cosine_similarity(embeddings_matrix, [query_embedding]).flatten()
    
    # Get top results
    top_indices = similarities.argsort()[::-1][:top_k]
    results = df.iloc[top_indices].copy()
    results['similarity'] = similarities[top_indices]
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 Video Content Q&A System</h1>', unsafe_allow_html=True)
    
    # Sidebar - simplified
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = ["llama3.2", "deepseek-r1"]
        selected_model = st.selectbox("Select Model", model_options, index=0)
        
        # Number of results
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        
        # Simple connection test
        if st.button("🔗 Test Ollama"):
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    st.success("✅ Ollama is running!")
                else:
                    st.error("❌ Ollama not responding")
            except:
                st.error("❌ Make sure Ollama is running: ollama serve")
    
    # Load embeddings
    with st.spinner("Loading embeddings..."):
        df = load_embeddings()
    
    if df is None:
        st.stop()
    
    # Display dataset info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Total Chunks", len(df))
    with col2:
        unique_videos = df['title'].nunique() if 'title' in df.columns else 0
        st.metric("🎬 Unique Videos", unique_videos)
    with col3:
        avg_similarity = 0  # Will be updated after query
        st.metric("🎯 Avg Similarity", f"{avg_similarity:.3f}")
    
    # Main query interface
    st.header("💬 Ask Your Question")
    
    # Query input
    query = st.text_area(
        "Enter your question about the video content:",
        placeholder="e.g., How do I create variables in Python?",
        height=100
    )
    
    # Search button
    col1, col2 = st.columns([1, 4])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary")
    
    if search_clicked and query.strip():
        with st.spinner("Searching for relevant content..."):
            # Search for similar content
            results = search_similar_content(query, df, top_k)
            
            if results is not None and not results.empty:
                # Update average similarity metric
                avg_sim = results['similarity'].mean()
                col3.metric("🎯 Avg Similarity", f"{avg_sim:.3f}")
                
                # Prepare context for LLM
                context_data = results[["title", "number", "start", "end", "text"]].to_json(orient="records")
                
                prompt = f'''I am teaching Python in my Data Science course. Here are video subtitle chunks containing video title, video number, start time in seconds, end time in seconds, the text at that time:

{context_data}
---------------------------------
"{query}"
User asked this question related to the video chunks, you have to answer in a human way (dont mention the above format, its just for you) where and how much content is taught in which video (in which video and at what timestamp) and guide the user to go to that particular video. If user asks unrelated question, tell him that you can only answer questions related to the course'''

                # Generate response
                with st.spinner("Generating response..."):
                    ai_response = generate_response(prompt, selected_model)
                
                if ai_response:
                    # Display AI response
                    st.header("🤖 AI Response")
                    st.markdown(ai_response)
                    
                    # Display relevant chunks
                    st.header("📋 Relevant Content Chunks")
                    
                    for idx, row in results.iterrows():
                        with st.expander(f"📹 {row.get('title', 'Unknown')} - Video {row.get('number', 'N/A')} (Similarity: {row['similarity']:.3f})"):
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.markdown(f"**Content:** {row.get('text', 'No text available')}")
                            
                            with col2:
                                start_time = format_timestamp(row.get('start', 0))
                                end_time = format_timestamp(row.get('end', 0))
                                st.markdown(f"""
                                <div class="timestamp">
                                    ⏰ {start_time} - {end_time}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Download options
                    st.header("💾 Download Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download AI response
                        st.download_button(
                            label="📄 Download AI Response",
                            data=ai_response,
                            file_name=f"ai_response_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    with col2:
                        # Download search results
                        csv_data = results[['title', 'number', 'start', 'end', 'text', 'similarity']].to_csv(index=False)
                        st.download_button(
                            label="📊 Download Search Results (CSV)",
                            data=csv_data,
                            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
            else:
                st.warning("⚠️ No results found. Please check your Ollama connection.")
    
    elif search_clicked:
        st.warning("⚠️ Please enter a question before searching.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #6b7280;'>
            🎓 Python For Data Science Course Q&A System<br>
            Powered by Ollama and Streamlit
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
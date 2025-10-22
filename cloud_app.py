# This is a frontend Streamlit app for a Video Content Q&A system.

import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import joblib
import requests
import os
from sentence_transformers import SentenceTransformer

# Page config
st.set_page_config(page_title="Video Q&A", page_icon="🎥", layout="wide")

# Title
st.title("🎥 Video Content Q&A System")
st.write("Ask questions about your video course content!")

@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings"""
    try:
        # Use a lightweight model that works well for search
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model
    except:
        st.error("Error loading embedding model")
        return None

@st.cache_data
def load_data():
    """Load the embeddings data"""
    try:
        df = joblib.load('embeddings.joblib')
        st.success(f"✅ Loaded {len(df)} video chunks")
        return df
    except:
        st.error("❌ embeddings.joblib not found.")
        return None

def create_embedding_cloud(text, model):
    """Create embedding using sentence-transformers"""
    try:
        embedding = model.encode([text])
        return embedding[0]
    except:
        st.error("Error creating embedding")
        return None

def get_response_cloud(prompt):
    """Get response using Hugging Face API (free tier)"""
    try:
        # Using Hugging Face Inference API (free tier)
        API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        
        # You can use this without token for limited requests
        # Or add your HF token in Streamlit secrets
        headers = {}
        if 'HF_TOKEN' in st.secrets:
            headers["Authorization"] = f"Bearer {st.secrets['HF_TOKEN']}"
        
        payload = {"inputs": prompt[:500]}  # Truncate for free tier
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response generated')
            return 'No response generated'
        else:
            # Fallback response if API fails
            return generate_fallback_response(prompt)
    except:
        return generate_fallback_response(prompt)

def generate_fallback_response(prompt):
    """Generate a simple fallback response when API fails"""
    # Extract video info from prompt
    try:
        import json
        # Find the JSON part in the prompt
        json_start = prompt.find('[{')
        json_end = prompt.rfind('}]') + 2
        
        if json_start != -1 and json_end != -1:
            json_data = prompt[json_start:json_end]
            data = json.loads(json_data)
            
            # Create simple response
            if data:
                first_chunk = data[0]
                video_title = first_chunk.get('title', 'Unknown')
                video_number = first_chunk.get('number', 'N/A')
                start_time = first_chunk.get('start', 0)
                
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                
                return f"""Based on your question, I found relevant content in:

**Video {video_number}: {video_title}**
- Timestamp: {minutes}:{seconds:02d}
- This video contains information related to your query.

Please watch the video from the mentioned timestamp to get detailed information about your question."""
            
    except:
        pass
    
    return "I found some relevant content in your course videos. Please check the related content sections below for specific videos and timestamps."

# Check if we're running locally or in the cloud
LOCAL_MODE = os.path.exists('embeddings.joblib') and 'localhost' not in os.environ.get('STREAMLIT_SERVER_ADDRESS', '')

if LOCAL_MODE:
    st.info("🏠 Running in Local Mode with Ollama")
    
    def create_embedding_local(text):
        try:
            r = requests.post("http://localhost:11434/api/embed", json={
                "model": "bge-m3",
                "input": [text]
            })
            return r.json()["embeddings"][0]
        except:
            st.error("❌ Ollama not running. Start with: ollama serve")
            return None

    def get_response_local(prompt):
        try:
            r = requests.post("http://localhost:11434/api/generate", json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            })
            return r.json()["response"]
        except:
            st.error("❌ Error getting response from Ollama")
            return None
else:
    st.info("☁️ Running in Cloud Mode")

# Load models/data
if LOCAL_MODE:
    df = load_data()
    embedding_model = None
else:
    # For cloud deployment, you might need to recreate embeddings with sentence-transformers
    # or provide a converted version
    st.warning("⚠️ Cloud mode requires pre-computed embeddings compatible with sentence-transformers")
    df = load_data()
    embedding_model = load_embedding_model()

if df is not None:
    # Show stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📊 Total Chunks", len(df))
    with col2:
        st.metric("🎬 Videos", df['title'].nunique())
    
    st.markdown("---")
    
    # Query input
    query = st.text_area("💬 Ask your question:", 
                        placeholder="e.g., How do I create variables in Python?",
                        height=100)
    
    if st.button("🔍 Get Answer", type="primary") and query:
        with st.spinner("Searching..."):
            # Get query embedding
            if LOCAL_MODE:
                query_embedding = create_embedding_local(query)
            else:
                if embedding_model:
                    query_embedding = create_embedding_cloud(query, embedding_model)
                else:
                    st.error("Embedding model not available")
                    query_embedding = None
            
            if query_embedding is not None:
                # Find similar content
                try:
                    embeddings_matrix = np.vstack(df['embedding'].values)
                    similarities = cosine_similarity(embeddings_matrix, [query_embedding]).flatten()
                    
                    # Get top 5 results
                    top_indices = similarities.argsort()[::-1][:5]
                    results = df.iloc[top_indices]
                    
                    # Create prompt
                    context = results[["title", "number", "start", "end", "text"]].to_json(orient="records")
                    
                    prompt = f'''I am teaching web development in my Sigma web development course. Here are video subtitle chunks:

{context}
---------------------------------
"{query}"
User asked this question related to the video chunks. Answer in a human way where and how much content is taught in which video (with video number and timestamp) and guide the user to go to that particular video. If unrelated question, tell them you can only answer course-related questions.'''

                    # Get AI response
                    with st.spinner("Generating answer..."):
                        if LOCAL_MODE:
                            response = get_response_local(prompt)
                        else:
                            response = get_response_cloud(prompt)
                    
                    if response:
                        # Show response
                        st.markdown("## 🤖 Answer")
                        st.write(response)
                        
                        # Show source chunks
                        st.markdown("## 📋 Related Content")
                        for idx, row in results.iterrows():
                            similarity_score = similarities[top_indices[list(results.index).index(idx)]]
                            with st.expander(f"📹 {row['title']} - Video {row['number']} ({similarity_score:.3f} similarity)"):
                                st.write(f"**Text:** {row['text']}")
                                st.write(f"**Time:** {int(row['start']//60)}:{int(row['start']%60):02d} - {int(row['end']//60)}:{int(row['end']%60):02d}")
                except Exception as e:
                    st.error(f"Error processing embeddings: {str(e)}")
                    st.info("This might be due to incompatible embedding formats between local and cloud modes.")
    
    # Instructions
    st.markdown("---")
    if LOCAL_MODE:
        st.markdown("""
        ### 📝 Local Mode Instructions:
        1. Make sure Ollama is running: `ollama serve`
        2. Make sure you have the models: `ollama pull bge-m3` and `ollama pull llama3.2`
        3. Ask any question about your course content
        4. Get answers with exact video timestamps!
        """)
    else:
        st.markdown("""
        ### ☁️ Cloud Mode Info:
        - Using Hugging Face API for responses (free tier with limits)
        - Embeddings computed with sentence-transformers
        - May have response limitations compared to local mode
        """)

else:
    st.markdown("""
    ### 🚀 Setup Instructions:
    1. Make sure `embeddings.joblib` file is available
    2. For local mode: Start Ollama and install models
    3. For cloud mode: Embeddings should be compatible with sentence-transformers
    4. Refresh this page
    """)
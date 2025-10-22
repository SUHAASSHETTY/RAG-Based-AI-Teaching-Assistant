# Local mode : BGE-M3 (Ollama) :: Cloud mode : Sentence-Transformers

# This code creates embeddings using sentence-transformers for cloud deployment because some cloud platforms may not support certain libraries like OpenAI's API or Faiss. The embeddings are saved in a joblib file for later use in a cloud application.

import os
import json
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer

def create_cloud_embeddings():
    """Create embeddings using sentence-transformers for cloud deployment"""
    
    # Load the model
    print("Loading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Process JSON files
    jsons = os.listdir("jsons")
    my_dicts = []
    chunk_id = 0
    
    for json_file in jsons:
        if json_file.endswith('.json'):
            print(f"Processing {json_file}...")
            
            with open(f"jsons/{json_file}", encoding='utf-8') as f:
                content = json.load(f)
            
            chunk_texts = [c['text'] for c in content['chunks']]
            
            if chunk_texts:
                # Create embeddings with sentence-transformers
                embeddings = model.encode(chunk_texts)
                
                for i, chunk in enumerate(content['chunks']):
                    chunk['chunk_id'] = chunk_id
                    chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization
                    chunk_id += 1
                    my_dicts.append(chunk)
    
    # Create DataFrame and save
    df = pd.DataFrame.from_records(my_dicts)
    print(f"Created DataFrame with {len(df)} chunks")
    
    # Convert embeddings back to numpy arrays
    df['embedding'] = df['embedding'].apply(lambda x: np.array(x))
    
    # Save
    joblib.dump(df, 'embeddings_cloud.joblib')
    print("Saved cloud-compatible embeddings to embeddings_cloud.joblib")

if __name__ == "__main__":
    import numpy as np
    create_cloud_embeddings()
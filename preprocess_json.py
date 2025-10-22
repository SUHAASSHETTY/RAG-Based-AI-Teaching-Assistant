# This script reads JSON files containing text chunks, generates embeddings for each chunk using a local embedding service, and saves the results in a DataFrame.

# Preprocess JSON files to create embeddings and save them in a DataFrame

import requests
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def create_embedding(text_list):
    try:
        # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
        r = requests.post("http://localhost:11434/api/embed", json={
            "model": "bge-m3",
            "input": text_list
        })
        r.raise_for_status()
        embedding = r.json()["embeddings"]
        return embedding
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

jsons = os.listdir("jsons")  # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    if json_file.endswith('.json'):
        try:
            with open(f"jsons/{json_file}", encoding='utf-8') as f:
                content = json.load(f)
            
            print(f"Creating Embeddings for {json_file}")
            
            chunk_texts = [c['text'] for c in content['chunks']]
            if not chunk_texts:
                continue
                
            embeddings = create_embedding(chunk_texts)
            if embeddings is None:
                continue
                   
            for i, chunk in enumerate(content['chunks']):
                chunk['chunk_id'] = chunk_id
                chunk['embedding'] = embeddings[i]
                chunk_id += 1
                my_dicts.append(chunk)
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue

if my_dicts:
    df = pd.DataFrame.from_records(my_dicts)
    print(f"Created DataFrame with {len(df)} chunks")
    
    # Save this dataframe
    joblib.dump(df, 'embeddings.joblib')
    print("Saved embeddings to embeddings.joblib")
else:
    print("No data to save")
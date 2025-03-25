import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os
import requests
def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
# Download large files at startup
download_file("https://drive.google.com/uc?export=download&id=1FF0Lx3rmop54LJiRvsfUVkPODROEIVXU", "embeddings.npy")
download_file("https://drive.google.com/uc?export=download&id=1yk-LHEV6-Z8o1zmLKBIxYZns6K9_LuR8", "patent_index.faiss")
download_file("https://drive.google.com/uc?export=download&id=19rOOm_zqC4ZNejKhO1sWMoSu1V_Krce2", "patents.csv")
download_file("https://drive.google.com/uc?export=download&id=1UuI1M0OtaU9JXSE9wBtl3_Ae4jOsHMOB", "preprocessed_patents.csv")

print("Loading preprocessed_patents.csv...")
df = pd.read_csv('preprocessed_patents.csv')
print(f"Loaded {len(df)} patents.")

print("Initializing SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model initialized.")

print("Generating embeddings...")
embeddings = model.encode(df['full_text'].tolist(), batch_size=32, show_progress_bar=True)
np.save('embeddings.npy', embeddings)
print(f"Embeddings saved to 'embeddings.npy'. Shape: {embeddings.shape}")

print("Building Faiss index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, 'patent_index.faiss')
print("Faiss index saved to 'patent_index.faiss'.")

print("Training complete!")
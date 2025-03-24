import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

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
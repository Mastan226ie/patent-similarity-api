import pandas as pd
import numpy as np
import faiss
import os
import gdown
from sentence_transformers import SentenceTransformer

def download_file(url, filename):
    if not os.path.exists(filename):
        try:
            print(f"Downloading {filename}...")
            gdown.download(url, filename, quiet=False)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")
            return False
    return True

# Download necessary files
file_urls = {
    "embeddings.npy": "https://drive.google.com/uc?id=1FF0Lx3rmop54LJiRvsfUVkPODROEIVXU",
    "patent_index.faiss": "https://drive.google.com/uc?id=1yk-LHEV6-Z8o1zmLKBIxYZns6K9_LuR8",
    "patents.csv": "https://drive.google.com/uc?id=19rOOm_zqC4ZNejKhO1sWMoSu1V_Krce2",
    "preprocessed_patents.csv": "https://drive.google.com/uc?id=1UuI1M0OtaU9JXSE9wBtl3_Ae4jOsHMOB"
}

for filename, url in file_urls.items():
    download_file(url, filename)

# Load dataset
filename = "preprocessed_patents.csv"
if os.path.exists(filename) and os.path.getsize(filename) > 0:
    df = pd.read_csv(filename)
    print(f"Loaded {len(df)} patents.")
else:
    print(f"Error: {filename} is missing or empty.")
    exit(1)

# Initialize SentenceTransformer
print("Initializing SentenceTransformer...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model initialized.")

# Generate and save embeddings in smaller batches to prevent memory overflow
print("Generating embeddings...")
batch_size = 10000  # Adjust batch size based on available memory
embeddings = model.encode(df['full_text'].tolist(), batch_size=32, show_progress_bar=True)
np.save('embeddings.npy', embeddings[:batch_size])
print(f"Saved {batch_size} embeddings to 'embeddings.npy'. Shape: {embeddings.shape}")

# Build and save Faiss index
print("Building Faiss index...")
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings[:batch_size])  # Add only first batch
faiss.write_index(index, 'patent_index.faiss')
print("Faiss index saved to 'patent_index.faiss'.")

print("Training complete!")

import os
import io
import numpy as np
import pandas as pd
import requests
import faiss
import nltk
import string
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw, ImageFont
import PyPDF2
import pdfplumber
from nltk.corpus import stopwords

# Ensure NLTK dependencies are available
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Patent Similarity API",
              description="API for patent similarity, plagiarism detection, and watermarking")

# Set up model and FAISS index
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

FILES = {
    "embeddings.npy": "1FF0Lx3rmop54LJiRvsfUVkPODROEIVXU",
    "patent_index.faiss": "1yk-LHEV6-Z8o1zmLKBIxYZns6K9_LuR8",
    "patents.csv": "19rOOm_zqC4ZNejKhO1sWMoSu1V_Krce2",
    "preprocessed_patents.csv": "1UuI1M0OtaU9JXSE9wBtl3_Ae4jOsHMOB",
}


def download_file(file_id, filename):
    """Download files if not already present (to avoid redownloading on every startup)."""
    if not os.path.exists(filename):
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        print(f"Downloading {filename}...")
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"Downloaded {filename}")
        except requests.RequestException as e:
            print(f"Failed to download {filename}: {e}")


# Download required files
for filename, file_id in FILES.items():
    download_file(file_id, filename)

# Load FAISS Index
index = faiss.read_index("patent_index.faiss")
embeddings = np.load("embeddings.npy")
patents_df = pd.read_csv("patents.csv")
preprocessed_patents_df = pd.read_csv("preprocessed_patents.csv")

print(f"Loaded embeddings with shape: {embeddings.shape}")
print(f"Loaded FAISS index with {index.ntotal} vectors")
print(f"Loaded patents DataFrame with {len(patents_df)} rows")
print(f"Loaded preprocessed patents DataFrame with {len(preprocessed_patents_df)} rows")


@app.post("/find_similar")
async def find_similar(text: str = Form(...)):
    query_embedding = model.encode([text])[0]
    k = 5
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    similar_patents = preprocessed_patents_df.iloc[indices[0]][['Patent ID', 'Title', 'Assignee']].to_dict('records')
    return {"similar_patents": similar_patents}


@app.post("/watermark")
async def watermark_document(
        file: UploadFile = File(...),
        watermark_type: str = Form(...),
        watermark_content: str = Form(None),
        output_filename: str = Form(...)
):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    input_pdf = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
    output_pdf = PyPDF2.PdfWriter()

    for page_num in range(len(input_pdf.pages)):
        page = input_pdf.pages[page_num]
        width, height = page.mediabox.upper_right
        watermark_img = Image.new('RGBA', (int(width), int(height)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark_img)

        if watermark_type == 'text':
            try:
                font = ImageFont.truetype("arial.ttf", 50)
            except:
                font = ImageFont.load_default()
            draw.text((50, 50), watermark_content, font=font, fill=(0, 0, 0, 100))

        watermark_bytes = io.BytesIO()
        watermark_img.convert('RGB').save(watermark_bytes, format='PDF')
        watermark_bytes.seek(0)
        watermark_pdf = PyPDF2.PdfReader(watermark_bytes)
        watermark_page = watermark_pdf.pages[0]
        page.merge_page(watermark_page)
        output_pdf.add_page(page)

    output_path = f"/tmp/{output_filename}.pdf"
    with open(output_path, 'wb') as f:
        output_pdf.write(f)
    return FileResponse(output_path, filename=output_filename, media_type='application/pdf')


@app.get("/")
async def root():
    return {"message": "Patent Similarity API is running"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

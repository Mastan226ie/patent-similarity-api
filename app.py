#import transformers
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import PyPDF2
import io
import os
import pdfplumber
import nltk
from nltk.corpus import stopwords
import string
import uvicorn
import nest_asyncio
import asyncio

nest_asyncio.apply()

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = FastAPI(title="Patent Similarity API", description="API for patent similarity, plagiarism detection, and watermarking")

model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index('patent_index.faiss')
embeddings = np.load('embeddings.npy')
df = pd.read_csv('preprocessed_patents.csv')

@app.post("/find_similar", response_model=dict)
async def find_similar(text: str = Form(...)):
    query_embedding = model.encode([text])[0]
    k = 5
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    similar_patents = df.iloc[indices[0]][['Patent ID', 'Title', 'Assignee']].to_dict('records')
    return {"similar_patents": similar_patents}

@app.post("/watermark", response_class=FileResponse)
async def watermark_document(
    file: UploadFile = File(...),
    watermark_type: str = Form(...),
    watermark_content: str = Form(None),
    output_filename: str = Form(...)
):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if watermark_type not in ['text', 'image']:
        raise HTTPException(status_code=400, detail="Watermark type must be 'text' or 'image'")
    if watermark_type == 'text' and not watermark_content:
        raise HTTPException(status_code=400, detail="Watermark text is required")

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
            text_bbox = draw.textbbox((0, 0), watermark_content, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            x = (int(width) - text_width) // 2
            y = (int(height) - text_height) // 2
            draw.text((x, y), watermark_content, font=font, fill=(0, 0, 0, 100))
        elif watermark_type == 'image':
            if not os.path.exists(watermark_content):
                raise HTTPException(status_code=400, detail="Watermark image file not found")
            logo = Image.open(watermark_content).convert('RGBA')
            logo = logo.resize((int(width // 3), int(height // 3)))
            x = (int(width) - logo.width) // 2
            y = (int(height) - logo.height) // 2
            watermark_img.paste(logo, (x, y), logo)

        watermark_bytes = io.BytesIO()
        watermark_img.convert('RGB').save(watermark_bytes, format='PDF')
        watermark_bytes.seek(0)
        watermark_pdf = PyPDF2.PdfReader(watermark_bytes)
        watermark_page = watermark_pdf.pages[0]
        page.merge_page(watermark_page)
        output_pdf.add_page(page)

    output_path = f"{output_filename}.pdf"
    with open(output_path, 'wb') as f:
        output_pdf.write(f)
    return FileResponse(output_path, filename=output_path, media_type='application/pdf')

@app.post("/check_plagiarism", response_model=dict)
async def check_plagiarism(file: UploadFile = File(...)):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""

    if not text.strip():
        raise HTTPException(status_code=400, detail="No text extracted from PDF")

    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    cleaned_text = ' '.join(tokens)

    query_embedding = model.encode([cleaned_text])[0]
    k = 1
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    closest_distance = distances[0][0]
    closest_patent = df.iloc[indices[0][0]][['Patent ID', 'Title', 'Assignee']].to_dict()

    SIMILARITY_THRESHOLD = 0.9
    if closest_distance < SIMILARITY_THRESHOLD:
        alert_message = (
            f"Alert: This document closely matches an existing patent!\n"
            f"Patent ID: {closest_patent['Patent ID']}\n"
            f"Title: {closest_patent['Title']}\n"
            f"Assignee: {closest_patent['Assignee']}\n"
            f"Similarity Distance: {closest_distance:.2f}"
        )
        return {"alert": alert_message, "is_plagiarized": True, "closest_match": closest_patent}
    return {"alert": "No significant matches found.", "is_plagiarized": False, "closest_match": closest_patent}

@app.get("/")
async def root():
    return {"message": "Patent Similarity API is running"}

# Run the server in Jupyter
async def run_server():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
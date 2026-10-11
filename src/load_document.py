# src/load_document.py
import os
import time
import pytesseract
from pdf2image import convert_from_path
from sklearn.metrics.pairwise import cosine_similarity
from google import genai

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler-26.09.0\Library\bin"
MODEL_NAME = "gemini-flash-lite-latest"


def load_scanned_pdf(path: str) -> str:
    # OCR is slow, so save the text next to the PDF and reuse it on later runs
    txt_path = path + ".ocr.txt"
    if os.path.exists(txt_path):
        print(f"Loaded saved OCR text from {txt_path}")
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read()

    pages = convert_from_path(path, poppler_path=POPPLER_PATH)
    full_text = ""
    for i, page_image in enumerate(pages):
        full_text += pytesseract.image_to_string(page_image) + "\n"
        print(f"Processed page {i+1}/{len(pages)}")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap
    return chunks


def find_top_chunks(query: str, chunks: list[str], chunk_embeddings, model, k: int = 1):
    """Return the k most similar chunks as (chunk, score) pairs, best first."""
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_idx = similarities.argsort()[::-1][:k]
    return [(chunks[i], float(similarities[i])) for i in top_idx]


def generate_answer(query: str, context_chunks: list[str], max_retries: int = 3) -> str:
    client = genai.Client()
    context = "\n\n---\n\n".join(context_chunks)

    prompt = f"""Answer the question using ONLY the context below.
If the context doesn't contain enough information to answer, say so clearly — do not guess or use outside knowledge.

Context:
{context}

Question: {query}

Answer:"""

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
            return response.text
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                return f"[Quota exhausted — try again later: {e}]"
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retry {attempt+1}/{max_retries} after error, waiting {wait}s...")
                time.sleep(wait)
            else:
                return f"[Generation failed after {max_retries} attempts: {e}]"
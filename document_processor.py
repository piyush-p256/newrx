import fitz  # PyMuPDF
import torch
from transformers import AutoTokenizer
from config import Config
import hashlib
import asyncio
from pathlib import Path
import aiohttp
import tempfile
import os
from urllib.parse import urlparse

# Detect device for acceleration
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
if DEVICE == "cpu":
    torch.set_num_threads(torch.get_num_threads())  # Optimize CPU threads

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

def content_hash(content: bytes) -> str:
    """Generate SHA-256 hash of content"""
    return hashlib.sha256(content).hexdigest()

async def parse_pdf(file_path: str) -> str:
    """Parse PDF using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        text = ""

        for page in doc:
            text += page.get_text("text", sort=True, flags=fitz.TEXT_DEHYPHENATE)

        return text
    except Exception as e:
        print(f"PDF parsing error: {e}")
        return ""

def chunk_text(text: str) -> list:
    """Chunk text into token-aware segments"""
    tokens = tokenizer.encode(text, return_tensors="pt").to(DEVICE)
    chunks = []

    for i in range(0, tokens.size(1), Config.CHUNK_SIZE - Config.CHUNK_OVERLAP):
        chunk_tokens = tokens[:, i:i+Config.CHUNK_SIZE]
        chunk_text = tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
        chunks.append(chunk_text)

    return chunks

async def download_to_tempfile(url: str) -> str:
    """Download remote file to temporary local file"""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                raise Exception(f"Failed to download file: {url}")
            data = await resp.read()

    suffix = ".pdf" if url.lower().endswith(".pdf") else ".txt"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(data)
    tmp.close()
    return tmp.name


async def process_document(file_path: str) -> tuple:
    """
    Process document from either a local path or a remote URL.
    Returns (hash, [text chunks])
    """
    is_url = urlparse(file_path).scheme in ("http", "https")
    temp_path = None

    try:
        if is_url:
            temp_path = await download_to_tempfile(file_path)
            file_path = temp_path

        content = Path(file_path).read_bytes()
        doc_hash = content_hash(content)

        if file_path.endswith(".pdf"):
            text = await parse_pdf(file_path)
        else:
            text = content.decode(errors="ignore")

        return doc_hash, chunk_text(text)

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception as e:
                print(f"Failed to delete temp file: {temp_path}, error: {e}")

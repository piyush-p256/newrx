import torch
from sentence_transformers import SentenceTransformer
from config import Config

# GPU-accelerated embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

def generate_embeddings(texts: list) -> list:
    """Batch generate embeddings with GPU acceleration"""
    return model.encode(
        texts,
        batch_size=Config.BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True
    ).tolist()
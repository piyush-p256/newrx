import os
import logging
from pinecone import Pinecone, ServerlessSpec
from config import Config

logger = logging.getLogger(__name__)

# Create a global Pinecone client instance
pc = Pinecone(api_key=Config.PINECONE_API_KEY)

def init_pinecone():
    try:
        if Config.PINECONE_INDEX not in pc.list_indexes().names():
            logger.info("Creating new Pinecone index...")
            pc.create_index(
                name=Config.PINECONE_INDEX,
                dimension=Config.EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=Config.PINECONE_ENVIRONMENT  # likely 'us-east-1'
                )
            )
        else:
            logger.info("Using existing Pinecone index")
    except Exception as e:
        logger.error(f"Pinecone initialization failed: {e}")
        raise

def get_index():
    try:
        return pc.Index(Config.PINECONE_INDEX)
    except Exception as e:
        logger.error(f"Failed to get Pinecone index: {e}")
        return None

def upsert_vectors(vectors: list):
    index = get_index()
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(
            vectors=batch,
            namespace=Config.PINECONE_NAMESPACE
        )

def query_index(query_vector: list, top_k=5) -> list:
    index = get_index()
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        namespace=Config.PINECONE_NAMESPACE,
        include_metadata=True
    )
    return [
        (match.metadata["text"], match.score)
        for match in results.matches
    ]

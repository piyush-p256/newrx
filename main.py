from fastapi import FastAPI, Depends, HTTPException, Header
from contextlib import asynccontextmanager
from pydantic import BaseModel
from document_processor import process_document
from embedding_service import generate_embeddings
from pinecone_service import init_pinecone, get_index, upsert_vectors, query_index
from llm_service import generate_response
from cache import init_db, is_processed, save_processing
from config import Config
import asyncio
import uuid
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models
class DocumentRequest(BaseModel):
    documents: list[str] | str
    questions: list[str]

class APIResponse(BaseModel):
    answers: list[str]

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup events
    logger.info("Initializing database...")
    init_db()
    
    logger.info("Initializing Pinecone...")
    init_pinecone()
    
    logger.info("Warming up GPU...")
    generate_embeddings(["warmup"])  # Initialize GPU model
    
    yield  # App runs here
    
    # Shutdown events
    logger.info("Cleaning up resources...")
    pinecone_index = get_index()
    if pinecone_index:
        pinecone_index.delete(delete_all=True, namespace=Config.PINECONE_NAMESPACE)

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Auth middleware
def verify_token(authorization: str = Header(...)):
    token = "82f5af99c6ce321fdbd4196aabc8f25feef8593924eb979ec060644672dca027"
    if not authorization.startswith(f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="Unauthorized")

# Processing pipeline
async def process_documents(documents: list):
    processed_data = []
    for doc in documents:
        doc_hash, chunks = await process_document(doc)
        
        if is_processed(doc_hash):
            logger.info(f"Document {doc_hash[:8]}... already processed")
            continue
        
        # Generate embeddings in batches
        vectors = []
        for i in range(0, len(chunks), Config.BATCH_SIZE):
            batch = chunks[i:i+Config.BATCH_SIZE]
            embeddings = generate_embeddings(batch)
            
            for j, embedding in enumerate(embeddings):
                vector_id = f"{doc_hash}_{i+j}"
                vectors.append((vector_id, embedding, {"text": batch[j], "doc_hash": doc_hash}))
        
        # Save to Pinecone and cache
        upsert_vectors(vectors)
        save_processing(doc_hash, [v[0] for v in vectors])
        logger.info(f"Processed {len(vectors)} chunks from {doc_hash[:8]}...")
        
        processed_data.append((doc_hash, chunks))
    return processed_data

# API Endpoint
@app.post("/hackrx/run", response_model=APIResponse)
async def run_query(request: DocumentRequest, auth: None = Depends(verify_token)):
    # Normalize documents to a list
    documents = request.documents if isinstance(request.documents, list) else [request.documents]

    # Process documents (with caching)
    logger.info(f"Processing {len(documents)} documents...")
    await process_documents(documents)

    # Process queries
    answers = []
    logger.info(f"Processing {len(request.questions)} questions...")

    for query in request.questions:
        # Generate query embedding
        query_embed = generate_embeddings([query])[0]

        # Retrieve relevant chunks
        context_chunks = query_index(query_embed)
        context = "\n\n".join([text for text, _ in context_chunks])
        logger.debug(f"Retrieved {len(context_chunks)} context chunks for query")

        # Generate plain answer
        answer = generate_response(query, context)
        if not answer:
            answer = "No valid response generated."
        answers.append(answer)

    return APIResponse(answers=answers)
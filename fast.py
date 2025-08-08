import os
import hashlib
import re
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import List
from mistralai.client import MistralClient

app = FastAPI()

# Create directories if they don't exist
os.makedirs("pdf", exist_ok=True)
os.makedirs("text_cache", exist_ok=True)

# Mistral API configuration
MISTRAL_API_KEY = "UncUfGAq0sBDW5Uvx9tSEV5FcCKjva2w"
client = MistralClient(api_key=MISTRAL_API_KEY)

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

def get_content_hash(content: bytes) -> str:
    """Generate SHA256 hash for content"""
    return hashlib.sha256(content).hexdigest()

def extract_text_from_pdf_url(url: str) -> str:
    # Download the PDF
    response = requests.get(url)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail="Unable to download document.")
    
    content = response.content
    
    # Create content-based hash
    content_hash = get_content_hash(content)
    pdf_path = f"pdf/{content_hash}.pdf"
    text_cache_path = f"text_cache/{content_hash}.txt"
    
    # Save PDF if not already saved
    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(content)
    
    # Check if text extraction is cached
    if os.path.exists(text_cache_path):
        with open(text_cache_path, "r", encoding="utf-8") as f:
            return f.read()
    
    # Extract text if not cached
    text = ""
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    
    # Save extracted text to cache
    with open(text_cache_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    return text

def query_llm(context: str, questions: List[str]) -> List[str]:
    # Enhanced prompt for concise answers
    prompt_template = (
        "You are a helpful assistant. Based strictly on the insurance policy document, "
        "provide clear, precise, and concise answers (1-2 sentences max) to the questions. "
        "Start answers with 'Yes' or 'No' when possible, followed by a brief explanation. "
        "Express all time periods in numerical form (e.g., '30 days' instead of 'thirty days'). "
        "Avoid bullet points, numbered sub-points, and policy-specific names.\n\n"
        "Document:\n{context}\n\nQuestions:\n"
    )

    full_prompt = prompt_template.format(context=context)
    for i, q in enumerate(questions, 1):
        full_prompt += f"{i}. {q}\n"
    
    # Explicit output format instruction
    full_prompt += "\nProvide ONLY the answers in a numbered list (1., 2., ...) without any additional text."

    response = client.chat(
        model="mistral-large-latest",
        messages=[{"role": "user", "content": full_prompt}]
    )

    raw_answer = response.choices[0].message.content.strip()
    
    # Improved parsing logic
    answers = []
    lines = raw_answer.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Extract answer text after number
        if re.match(r'^\d+[\.\)]\s*', line):  # Handles 1., 2), etc.
            answer = re.sub(r'^\d+[\.\)]\s*', '', line)
            answers.append(answer)
        # Handle answers without numbers
        elif len(answers) < len(questions):
            answers.append(line)
    
    # Ensure we have exactly one answer per question
    if len(answers) < len(questions):
        answers.extend([f"Answer not found for question {i+1}" 
                      for i in range(len(answers), len(questions))])
    elif len(answers) > len(questions):
        answers = answers[:len(questions)]
        
    return answers

@app.post("/hackrx/run", response_model=QueryResponse)
def run_policy_qa(
    payload: QueryRequest,
    authorization: str = Header(..., alias="Authorization")
):
    # Simple API key check (for demo purposes)
    expected_token = "82f5af99c6ce321fdbd4196aabc8f25feef8593924eb979ec060644672dca027"
    if not authorization.endswith(expected_token):
        raise HTTPException(status_code=401, detail="Unauthorized")

    # Extract and process
    document_text = extract_text_from_pdf_url(payload.documents)
    answers = query_llm(document_text, payload.questions)
    return {"answers": answers}
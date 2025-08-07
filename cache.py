import sqlite3
import hashlib
from config import Config

def init_db():
    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS processed_docs (
            doc_hash TEXT PRIMARY KEY,
            pinecone_ids TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def is_processed(doc_hash: str) -> bool:
    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM processed_docs WHERE doc_hash=?", (doc_hash,))
    exists = cursor.fetchone() is not None
    conn.close()
    return exists

def save_processing(doc_hash: str, pinecone_ids: list):
    conn = sqlite3.connect(Config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO processed_docs (doc_hash, pinecone_ids)
        VALUES (?, ?)
    """, (doc_hash, ",".join(pinecone_ids)))
    conn.commit()
    conn.close()
import uuid
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from rag.embeddings import get_embeddings_model
from config import settings
from database.connection import SessionLocal
from database.models import Document, DocumentChunk
from sqlalchemy import text

def get_vector_store():
    embeddings = get_embeddings_model()
    # Ensure correct connection string for async/sync depending on langchain-postgres
    return PGVector(
        embeddings=embeddings,
        collection_name="voicepdf_collection",
        connection=settings.DATABASE_URL,
        use_jsonb=True,
    )

def index_chunks(document_id: str, filename: str, chunks: list):
    """Index chunks into PostgreSQL with pgvector."""
    vector_store = get_vector_store()
    
    # We can also add to our relational tracking table
    with SessionLocal() as db:
        # Create Document record
        doc_record = Document(id=document_id, filename=filename, status='indexed')
        db.add(doc_record)
        db.commit()
    
    # Let vector_store handle embeddings and insertions
    vector_store.add_documents(chunks)
    
    return len(chunks)

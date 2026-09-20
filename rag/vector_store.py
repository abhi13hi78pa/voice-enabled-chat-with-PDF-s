from langchain_postgres.vectorstores import PGVector
from rag.embeddings import get_embeddings_model
from config import settings
from database.connection import SessionLocal
from database.models import Document
from langchain_core.documents import Document as LC_Document

def get_vector_store() -> PGVector:
    """Initialize and return the PGVector store instance."""
    embeddings = get_embeddings_model()
    
    # LangChain PGVector uses the connection string to handle the pgvector integration.
    # We use a specific collection name for this project.
    return PGVector(
        embeddings=embeddings,
        collection_name="voicepdf_collection",
        connection=settings.DATABASE_URL,
        use_jsonb=True,
    )

def index_documents(document_id: str, filename: str, chunks: list[LC_Document]) -> int:
    """
    Generate embeddings for chunks and store them in PostgreSQL via pgvector.
    Preserves document and chunk metadata.
    Avoids duplicate indexing by checking if the document_id already exists.
    """
    if not chunks:
        raise ValueError("No chunks provided for indexing.")
        
    with SessionLocal() as db:
        # Check for duplicates to prevent re-indexing the same document blindly
        existing_doc = db.query(Document).filter(Document.id == document_id).first()
        if existing_doc and existing_doc.status == 'indexed':
            # Skip indexing if already present
            return 0
            
        # Create or update Document tracking record
        if not existing_doc:
            doc_record = Document(id=document_id, filename=filename, status='indexing')
            db.add(doc_record)
        else:
            existing_doc.status = 'indexing'
        db.commit()

    try:
        vector_store = get_vector_store()
        
        # Add documents to the vector store (this computes embeddings and stores them)
        vector_store.add_documents(chunks)
        
        # Update status
        with SessionLocal() as db:
            doc_record = db.query(Document).filter(Document.id == document_id).first()
            if doc_record:
                doc_record.status = 'indexed'
                db.commit()
                
        return len(chunks)
        
    except Exception as e:
        # Mark as failed in DB
        with SessionLocal() as db:
            doc_record = db.query(Document).filter(Document.id == document_id).first()
            if doc_record:
                doc_record.status = 'failed'
                db.commit()
        raise RuntimeError(f"Failed to index document chunks: {str(e)}")

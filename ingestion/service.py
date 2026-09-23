"""Ingestion Service — Orchestrates the complete PDF ingestion pipeline."""
import os
import uuid
import logging

from ingestion.pdf_loader import validate_pdf, load_pdf
from ingestion.chunker import chunk_documents
from rag.vector_store import index_documents
from config import settings

logger = logging.getLogger(__name__)


def ingest_pdf(file_path: str) -> dict:
    """
    Complete PDF ingestion pipeline: validate → load → chunk → index.
    
    Args:
        file_path: Path to the uploaded PDF file.
    
    Returns:
        dict with keys:
            - 'document_id': The generated document UUID
            - 'filename': Original filename
            - 'chunks_indexed': Number of chunks stored
            - 'status': 'success' or 'error'
            - 'message': Status message
    """
    filename = os.path.basename(file_path)
    document_id = str(uuid.uuid4())
    
    try:
        # Step 1: Validate
        if not validate_pdf(file_path):
            return {
                'document_id': document_id,
                'filename': filename,
                'chunks_indexed': 0,
                'status': 'error',
                'message': f'Invalid PDF file: {filename}'
            }
        
        # Step 2: Load pages
        logger.info(f"Loading PDF: {filename} (ID: {document_id})")
        pages = load_pdf(file_path, document_id)
        if not pages:
            return {
                'document_id': document_id,
                'filename': filename,
                'chunks_indexed': 0,
                'status': 'error',
                'message': f'No text content found in: {filename}'
            }
        
        # Step 3: Chunk
        logger.info(f"Chunking {len(pages)} pages from {filename}")
        chunks = chunk_documents(pages)
        if not chunks:
            return {
                'document_id': document_id,
                'filename': filename,
                'chunks_indexed': 0,
                'status': 'error',
                'message': f'No chunks generated from: {filename}'
            }
        
        # Step 4: Index (embed + store in pgvector)
        logger.info(f"Indexing {len(chunks)} chunks for {filename}")
        chunks_indexed = index_documents(document_id, filename, chunks)
        
        logger.info(f"Successfully indexed {chunks_indexed} chunks for {filename}")
        return {
            'document_id': document_id,
            'filename': filename,
            'chunks_indexed': chunks_indexed,
            'status': 'success',
            'message': f'Successfully indexed {chunks_indexed} chunks from {filename}'
        }
    
    except Exception as e:
        logger.error(f"Ingestion failed for {filename}: {e}")
        return {
            'document_id': document_id,
            'filename': filename,
            'chunks_indexed': 0,
            'status': 'error',
            'message': f'Ingestion failed for {filename}: {str(e)}'
        }

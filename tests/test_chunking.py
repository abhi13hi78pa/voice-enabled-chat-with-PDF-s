import pytest
from langchain_core.documents import Document
from ingestion.chunker import chunk_documents
from config import settings

def test_chunking_preserves_metadata():
    docs = [
        Document(
            page_content="This is a test document. " * 50,
            metadata={"document_id": "123", "page_number": 1}
        )
    ]
    
    settings.CHUNK_SIZE = 100
    settings.CHUNK_OVERLAP = 20
    
    chunks = chunk_documents(docs)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["document_id"] == "123"
        assert chunk.metadata["page_number"] == 1
        assert "chunk_index" in chunk.metadata

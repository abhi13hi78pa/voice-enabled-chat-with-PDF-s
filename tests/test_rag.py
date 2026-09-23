import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document as LC_Document
from rag.embeddings import get_embeddings_model
from rag.vector_store import index_documents
from rag.retriever import get_retriever
from config import settings

def test_embeddings_initialization_missing_openai_key():
    # Store old values
    old_key = settings.OPENAI_API_KEY
    old_provider = settings.EMBEDDING_PROVIDER
    settings.EMBEDDING_PROVIDER = "openai"
    settings.OPENAI_API_KEY = ""
    
    try:
        with pytest.raises(ValueError, match="OPENAI_API_KEY is missing"):
            get_embeddings_model()
    finally:
        # Restore old values
        settings.OPENAI_API_KEY = old_key
        settings.EMBEDDING_PROVIDER = old_provider

def test_embeddings_initialization_missing_nvidia_key():
    # Store old values
    old_key = settings.NVIDIA_API_KEY
    old_provider = settings.EMBEDDING_PROVIDER
    settings.EMBEDDING_PROVIDER = "nvidia"
    settings.NVIDIA_API_KEY = ""
    
    try:
        with pytest.raises(ValueError, match="NVIDIA_API_KEY is missing"):
            get_embeddings_model()
    finally:
        # Restore old values
        settings.NVIDIA_API_KEY = old_key
        settings.EMBEDDING_PROVIDER = old_provider

@patch("rag.embeddings.OpenAIEmbeddings")
def test_retriever_configuration(mock_embeddings_cls):
    """Test that retriever is configured with correct TOP_K and handles document filters."""
    # We mock get_vector_store to avoid DB connection in this unit test
    with patch("rag.retriever.get_vector_store") as mock_get_store:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store
        
        # Test without document IDs
        retriever = get_retriever()
        mock_store.as_retriever.assert_called_with(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K}
        )
        
        # Test with document IDs
        retriever_filtered = get_retriever(document_ids=["doc123"])
        mock_store.as_retriever.assert_called_with(
            search_type="similarity",
            search_kwargs={"k": settings.TOP_K, "filter": {"document_id": {"$in": ["doc123"]}}}
        )

@patch("rag.vector_store.SessionLocal")
@patch("rag.vector_store.get_vector_store")
def test_index_documents_preserves_metadata(mock_get_store, mock_session):
    """Test that index_documents correctly handles valid chunks and prevents duplicates."""
    mock_db = MagicMock()
    mock_session.return_value.__enter__.return_value = mock_db
    
    # Simulate that the document does not exist yet (no duplicate)
    mock_db.query.return_value.filter.return_value.first.return_value = None
    
    mock_store = MagicMock()
    mock_get_store.return_value = mock_store
    
    chunks = [LC_Document(page_content="test", metadata={"chunk_id": "1"})]
    
    num_indexed = index_documents("doc1", "file.pdf", chunks)
    
    assert num_indexed == 1
    mock_store.add_documents.assert_called_once_with(chunks)

def test_index_documents_empty_chunks():
    with pytest.raises(ValueError, match="No chunks provided"):
        index_documents("doc1", "file.pdf", [])

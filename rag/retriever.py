from rag.vector_store import get_vector_store
from config import settings

def get_retriever(document_ids: list[str] = None):
    """
    Returns a configured retriever for semantic similarity search.
    Optionally filters by specific document IDs.
    """
    try:
        vector_store = get_vector_store()
    except ValueError as e:
        # Pass up configuration errors (like missing API keys)
        raise RuntimeError(f"Retriever configuration error: {str(e)}")
    
    search_kwargs = {"k": settings.TOP_K}
    
    # Metadata filtering if specific documents are selected
    if document_ids and len(document_ids) > 0:
        # LangChain PGVector supports metadata filtering via dictionaries
        search_kwargs["filter"] = {"document_id": {"$in": document_ids}}
        
    # The retriever will return chunks containing their original metadata
    # (chunk_id, page_number, document_name, etc.)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs
    )
    return retriever

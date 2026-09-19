from ingestion.indexer import get_vector_store
from config import settings

def get_retriever(document_ids: list[str] = None):
    """Returns a retriever, optionally filtered by specific document IDs."""
    vector_store = get_vector_store()
    
    search_kwargs = {"k": settings.TOP_K}
    
    # Metadata filtering if specific documents are selected
    if document_ids and len(document_ids) > 0:
        # LangChain PGVector supports metadata filtering via dictionaries
        search_kwargs["filter"] = {"document_id": {"$in": document_ids}}
        
    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    return retriever

import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into chunks for embedding.
    Ensures that each chunk has a unique chunk_id and preserves necessary metadata.
    """
    if not documents:
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk-specific metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata["chunk_index"] = i
        
        # Ensure page_number is standardized. PyPDFLoader uses 'page'
        if "page" in chunk.metadata:
            chunk.metadata["page_number"] = chunk.metadata["page"] + 1 # 1-indexed for user display
        else:
            chunk.metadata["page_number"] = 1
            
        # Ensure source is standard
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = chunk.metadata.get("document_name", "Unknown")
        
    return chunks

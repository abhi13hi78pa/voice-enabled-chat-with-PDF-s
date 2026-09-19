from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import settings

def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        
    return chunks

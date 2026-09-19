import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf(file_path: str, document_id: str) -> list[Document]:
    """Load a PDF and return page-aware LangChain Documents."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at {file_path}")
        
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Enrich metadata
    filename = os.path.basename(file_path)
    for doc in docs:
        doc.metadata["document_id"] = document_id
        doc.metadata["document_name"] = filename
        
    return docs

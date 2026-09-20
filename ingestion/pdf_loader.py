import os
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def validate_pdf(file_path: str) -> bool:
    """Validate that the file exists, is a PDF, and is not completely empty/corrupt."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")
        
    if not file_path.lower().endswith('.pdf'):
        raise ValueError("File is not a PDF.")
        
    # Basic check for corrupt or completely empty files
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            if len(reader.pages) == 0:
                raise ValueError("PDF has no pages.")
    except Exception as e:
        raise ValueError(f"Invalid or corrupt PDF file: {str(e)}")
        
    return True

def load_pdf(file_path: str, document_id: str) -> list[Document]:
    """
    Load a PDF, extract text, and return page-aware LangChain Documents.
    Preserves page numbers and enriches metadata with document_id.
    """
    validate_pdf(file_path)
    
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    if not docs:
        raise ValueError("No extractable text found in the PDF.")
        
    # Enrich metadata
    filename = os.path.basename(file_path)
    
    valid_docs = []
    for doc in docs:
        # Some PDFs might have empty pages; we filter them out to avoid empty chunks
        if not doc.page_content.strip():
            continue
            
        doc.metadata["document_id"] = document_id
        doc.metadata["document_name"] = filename
        # PyPDFLoader usually stores the page number in metadata['page'] (0-indexed)
        # We ensure it's there.
        if "page" not in doc.metadata:
            doc.metadata["page"] = 0
            
        valid_docs.append(doc)
        
    if not valid_docs:
         raise ValueError("PDF contains no extractable text across its pages.")
         
    return valid_docs

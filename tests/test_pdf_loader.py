import os
import pytest
import tempfile
from ingestion.pdf_loader import validate_pdf, load_pdf

def test_validate_pdf_not_found():
    with pytest.raises(FileNotFoundError):
        validate_pdf("nonexistent_file.pdf")

def test_validate_pdf_wrong_extension():
    # Create a temporary txt file
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    
    with pytest.raises(ValueError, match="not a PDF"):
        validate_pdf(path)
        
    os.remove(path)

def test_validate_pdf_corrupt_file():
    # Create a dummy PDF with bad content
    fd, path = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, 'w') as f:
        f.write("This is not a real PDF.")
        
    with pytest.raises(ValueError, match="Invalid or corrupt PDF file"):
        validate_pdf(path)
        
    os.remove(path)

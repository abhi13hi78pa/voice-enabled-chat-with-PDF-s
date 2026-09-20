import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database.models import Base, Document, DocumentChunk
from ui.gradio_app import create_ui

def test_configuration_loads():
    """Verify that environment variables or defaults are loaded."""
    assert settings.PROJECT_NAME is not None
    assert settings.ENVIRONMENT is not None
    assert settings.DATABASE_URL is not None

def test_database_models_import():
    """Verify that DB models are syntactically valid and importable."""
    assert Document.__tablename__ == 'documents'
    assert DocumentChunk.__tablename__ == 'document_chunks'

def test_ui_foundation_initializes():
    """Verify that the Gradio UI blocks can be instantiated without crashing."""
    demo = create_ui()
    assert demo is not None

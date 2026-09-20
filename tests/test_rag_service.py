import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document as LC_Document
from config import settings
from rag.llm import get_llm
from rag.context import build_context_string, extract_citations
from rag.rag_service import generate_answer

def test_llm_initialization_missing_key():
    old_key = settings.OPENAI_API_KEY
    settings.OPENAI_API_KEY = ""
    with pytest.raises(ValueError, match="OPENAI_API_KEY is missing"):
        get_llm()
    settings.OPENAI_API_KEY = old_key

def test_context_builder():
    docs = [
        LC_Document(page_content="text 1", metadata={"source": "a.pdf", "page_number": 1}),
        LC_Document(page_content="text 2", metadata={"source": "a.pdf", "page_number": 1}),
        LC_Document(page_content="text 3", metadata={"source": "b.pdf", "page_number": 2})
    ]
    
    context = build_context_string(docs)
    assert "[Source: a.pdf | Page: 1]" in context
    assert "text 1" in context
    
    citations = extract_citations(docs)
    assert len(citations) == 2
    assert "a.pdf — Page 1" in citations
    assert "b.pdf — Page 2" in citations

@patch("rag.rag_service.get_retriever")
@patch("rag.rag_service.get_llm")
def test_generate_answer_orchestration(mock_get_llm, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_get_retriever.return_value = mock_retriever
    
    # Simulate retrieved docs
    mock_retriever.invoke.return_value = [
        LC_Document(page_content="Found information.", metadata={"source": "doc.pdf", "page_number": 5})
    ]
    
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    
    # Simulate LLM response
    mock_response = MagicMock()
    mock_response.content = "This is the generated answer."
    mock_llm.invoke.return_value = mock_response
    mock_llm.return_value = mock_response
    
    answer, citations = generate_answer("What is this?", document_ids=["123"])
    
    assert answer == "This is the generated answer."
    assert "doc.pdf — Page 5" in citations
    mock_retriever.invoke.assert_called_once_with("What is this?")

@patch("rag.rag_service.get_retriever")
def test_generate_answer_empty_retrieval(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_get_retriever.return_value = mock_retriever
    mock_retriever.invoke.return_value = []
    
    answer, citations = generate_answer("No docs match", document_ids=["123"])
    assert "not found" in answer.lower()
    assert len(citations) == 0

def test_generate_answer_empty_question():
    with pytest.raises(ValueError, match="Cannot generate an answer for an empty question"):
        generate_answer("   ", document_ids=["doc1"])

def test_generate_answer_no_documents():
    with pytest.raises(ValueError, match="No PDF documents have been indexed"):
        generate_answer("What is this?", document_ids=[])

@patch("rag.rag_service.get_retriever")
def test_generate_answer_retriever_failure(mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = RuntimeError("Database connection failed")
    mock_get_retriever.return_value = mock_retriever
    
    with pytest.raises(RuntimeError, match="Database connection failed"):
        generate_answer("What is this?", document_ids=["doc1"])

@patch("rag.rag_service.get_retriever")
@patch("rag.rag_service.get_llm")
def test_generate_answer_llm_failure(mock_get_llm, mock_get_retriever):
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        LC_Document(page_content="Context text", metadata={"source": "doc.pdf", "page_number": 1})
    ]
    mock_get_retriever.return_value = mock_retriever
    
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = RuntimeError("OpenAI API rate limit exceeded")
    mock_llm.side_effect = RuntimeError("OpenAI API rate limit exceeded")
    mock_get_llm.return_value = mock_llm
    
    with pytest.raises(RuntimeError, match="OpenAI API rate limit exceeded"):
        generate_answer("What is this?", document_ids=["doc1"])

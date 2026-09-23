"""
RAG Chain — Convenience wrapper around rag_service for backward compatibility.

This module provides `ask_question` as a simple interface to the full RAG pipeline
implemented in rag_service.generate_answer().
"""
from rag.rag_service import generate_answer


def ask_question(question: str, document_ids: list[str] = None) -> dict:
    """
    Ask a question against indexed documents and receive a grounded answer
    with source citations.

    Args:
        question: The user's question string.
        document_ids: Optional list of document IDs to filter retrieval scope.

    Returns:
        dict with keys:
            - "answer": The LLM-generated grounded answer string.
            - "citations": List of deduplicated source citation strings.
    """
    answer, citations = generate_answer(question, document_ids=document_ids)
    return {"answer": answer, "citations": citations}

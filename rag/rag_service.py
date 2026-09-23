"""
RAG Service — Orchestrates the end-to-end RAG pipeline with conversation memory support.

Phase 4: Grounded RAG generation with citations.
Phase 6: Multi-turn conversation memory integration.
"""
import logging

from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.prompts import get_rag_prompt
from rag.context import build_context_string, extract_citations

logger = logging.getLogger(__name__)


def _sanitize_llm_error(e: Exception) -> RuntimeError:
    """Map LLM exceptions to clear, user-friendly messages without exposing API keys or internals."""
    error_type = type(e).__name__
    msg = str(e).lower()

    if "authentication" in msg or "401" in msg or error_type == "AuthenticationError":
        return RuntimeError("NVIDIA API authentication failed: Invalid or expired API key. Please check your NVIDIA_API_KEY in .env.")
    elif "permission" in msg or "403" in msg or error_type == "PermissionDeniedError":
        return RuntimeError("NVIDIA API access unauthorized: You do not have permission to access the requested model.")
    elif "not found" in msg or "404" in msg or error_type == "NotFoundError":
        return RuntimeError("The requested NVIDIA model is unavailable or not found.")
    elif "rate limit" in msg or "429" in msg or error_type == "RateLimitError":
        return RuntimeError("NVIDIA API rate limit exceeded. Please wait a moment before trying again.")
    elif "timeout" in msg or error_type == "APITimeoutError":
        return RuntimeError("NVIDIA API request timed out. Please check your network connection and try again.")
    elif "connection" in msg or error_type == "APIConnectionError":
        return RuntimeError("Failed to connect to the NVIDIA API endpoint. Please check your network connection.")
    elif "bad request" in msg or "400" in msg or error_type == "BadRequestError":
        return RuntimeError("Malformed request or incompatible parameters sent to NVIDIA API.")
    else:
        return RuntimeError(f"LLM Generation Error: {str(e)}")


def generate_answer(
    question: str,
    document_ids: list[str] = None,
    chat_history: str = None,
) -> tuple[str, list[str]]:
    """
    Orchestrates the RAG pipeline with optional conversation memory:
    1. Retrieves relevant chunks for the question.
    2. Checks if chunks exist.
    3. Formats context and conversation history.
    4. Invokes LLM with grounded prompt.
    5. Returns the grounded answer and a list of citations.

    Args:
        question: The user's question string.
        document_ids: List of indexed document IDs to scope retrieval.
        chat_history: Optional formatted conversation history string
                      from ConversationMemory.get_context_string().

    Returns:
        Tuple of (answer_text, list_of_citation_strings).
    """
    if not question or not question.strip():
        raise ValueError("Cannot generate an answer for an empty question.")

    if not document_ids:
        # Require at least one document ID context to prevent searching an empty or entirely unrelated DB
        raise ValueError("No PDF documents have been indexed yet. Please upload and process a PDF first.")

    # 1. Retrieval
    retriever = get_retriever(document_ids=document_ids)
    docs = retriever.invoke(question)

    # 2. Check for empty retrieval
    if not docs:
        return (
            "I'm sorry, but no relevant information could be found in the provided documents.",
            []
        )

    # 3. Build Context & Citations
    context_str = build_context_string(docs)
    citations = extract_citations(docs)

    # 4. Prompt & LLM
    llm = get_llm()
    prompt = get_rag_prompt()

    # Prepare the conversation history (default to empty if not provided)
    history_str = chat_history if chat_history else "No previous conversation."

    # Using LCEL (LangChain Expression Language) for execution
    chain = prompt | llm

    try:
        response = chain.invoke({
            "context": context_str,
            "question": question,
            "chat_history": history_str,
        })
    except Exception as e:
        raise _sanitize_llm_error(e) from None

    # 5. Return grounded answer and citations
    return response.content, citations

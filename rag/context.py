from langchain_core.documents import Document

def build_context_string(docs: list[Document]) -> str:
    """
    Converts retrieved documents into a formatted text context for the LLM.
    Embeds metadata clearly so the LLM understands document boundaries.
    """
    if not docs:
        return "No context available."
        
    context_parts = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page_number", "?")
        
        chunk_header = f"[Source: {source} | Page: {page}]"
        context_parts.append(f"{chunk_header}\n{doc.page_content.strip()}")
        
    return "\n\n".join(context_parts)

def extract_citations(docs: list[Document]) -> list[str]:
    """
    Extracts a deduplicated list of human-readable citations from the retrieved documents.
    Prevents repeating the same page multiple times if several chunks came from it.
    """
    citations = set()
    for doc in docs:
        source = doc.metadata.get("source", "Unknown Document")
        page = doc.metadata.get("page_number", "?")
        citations.add(f"{source} — Page {page}")
        
    # Sort for deterministic UI presentation
    return sorted(list(citations))

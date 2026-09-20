from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.prompts import get_rag_prompt
from rag.context import build_context_string, extract_citations

def generate_answer(question: str, document_ids: list[str] = None) -> tuple[str, list[str]]:
    """
    Orchestrates the RAG pipeline:
    1. Retrieves relevant chunks for the question.
    2. Checks if chunks exist.
    3. Formats context.
    4. Invokes LLM.
    5. Returns the grounded answer and a list of citations.
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
    
    # Using LCEL (LangChain Expression Language) for execution
    chain = prompt | llm
    
    response = chain.invoke({
        "context": context_str,
        "question": question
    })
    
    # 5. Return grounded answer and citations
    return response.content, citations

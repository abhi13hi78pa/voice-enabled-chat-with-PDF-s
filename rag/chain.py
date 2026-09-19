from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from rag.prompts import qa_prompt
from rag.retriever import get_retriever
from config import settings
import json

def get_llm():
    if settings.LLM_PROVIDER.lower() == "openai":
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

def format_docs(docs):
    formatted = []
    for doc in docs:
        page = doc.metadata.get('page', doc.metadata.get('page_number', 'Unknown'))
        filename = doc.metadata.get('document_name', 'Unknown')
        content = doc.page_content
        formatted.append(f"Source: {filename}, Page: {page}\nContent: {content}\n")
    return "\n".join(formatted)

def create_rag_chain(document_ids=None):
    retriever = get_retriever(document_ids)
    llm = get_llm()
    
    # We will build a basic conversational RAG chain
    
    def contextualize(inputs):
        """Extract needed info."""
        return {
            "context": format_docs(inputs["docs"]),
            "question": inputs["question"],
            "chat_history": inputs.get("chat_history", [])
        }

    rag_chain = (
        {
            "docs": retriever,
            "question": RunnablePassthrough(),
            "chat_history": RunnablePassthrough() 
        }
        | RunnableLambda(lambda x: {"context": format_docs(x["docs"]), "question": x["question"], "chat_history": x.get("chat_history", [])})
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    # To properly support chat history and docs extraction, we might need a slightly different setup 
    # to return docs for citation. But for simplicity, we let the LLM generate citations based on format_docs.
    
    return rag_chain

def ask_question(question: str, chat_history: list, document_ids: list = None):
    """Entry point for asking a question."""
    chain = create_rag_chain(document_ids)
    # The chain expects a dict or the string itself if mapped properly. Let's fix the chain.
    
    retriever = get_retriever(document_ids)
    docs = retriever.invoke(question)
    context = format_docs(docs)
    
    llm = get_llm()
    messages = qa_prompt.format_messages(
        context=context,
        chat_history=chat_history,
        question=question
    )
    response = llm.invoke(messages)
    
    return response.content, docs

from langchain_openai import ChatOpenAI
from config import settings

def get_llm():
    """
    Initializes and returns the configured LLM for generation.
    Raises clear errors if credentials or configurations are missing.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.strip() == "" or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid. "
                "Please configure it in your .env file to use OpenAI chat models."
            )
            
        return ChatOpenAI(
            model_name=settings.LLM_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.0 # Strict grounded answering, no creativity needed
        )
    else:
        raise ValueError(f"Unsupported LLM provider configured: {provider}")

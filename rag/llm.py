from langchain_openai import ChatOpenAI
from config import settings

def get_llm():
    """
    Initializes and returns the configured LLM for generation.
    Configured for NVIDIA Nemotron via NVIDIA's OpenAI-compatible endpoint,
    with backwards-compatible support for OpenAI.
    Raises clear, secure errors if credentials or configurations are missing.
    """
    provider = settings.LLM_PROVIDER.lower()
    
    if provider == "nvidia":
        if (
            not settings.NVIDIA_API_KEY 
            or settings.NVIDIA_API_KEY.strip() == "" 
            or settings.NVIDIA_API_KEY == "your_nvidia_api_key_here"
        ):
            raise ValueError(
                "NVIDIA_API_KEY is missing or invalid. "
                "Please configure it in your .env file to use NVIDIA Nemotron."
            )
            
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.NVIDIA_API_KEY,
            base_url=settings.NVIDIA_BASE_URL,
            temperature=0.0,
            max_tokens=1024
        )
    elif provider == "openai":
        if (
            not settings.OPENAI_API_KEY 
            or settings.OPENAI_API_KEY.strip() == "" 
            or settings.OPENAI_API_KEY == "your_openai_api_key_here"
        ):
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid. "
                "Please configure it in your .env file to use OpenAI chat models."
            )
            
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported LLM provider configured: {provider}")

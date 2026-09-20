import os
from langchain_openai import OpenAIEmbeddings
from config import settings

def get_embeddings_model():
    """
    Returns the configured embeddings model based on the environment configuration.
    Raises clear errors if required credentials or configurations are missing.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()
    
    if provider == "openai":
        if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY.strip() == "" or settings.OPENAI_API_KEY == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid. "
                "Please configure it in your .env file to use OpenAI embeddings."
            )
            
        # The dimension should match the configured model (e.g. 1536 for text-embedding-3-small)
        # We rely on settings.EMBEDDING_DIMENSION to keep track of this on the DB side if needed.
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
    else:
        raise ValueError(f"Unsupported embedding provider configured: {provider}")

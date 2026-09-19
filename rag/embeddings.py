from langchain_openai import OpenAIEmbeddings
from config import settings

def get_embeddings_model():
    """Returns the configured embeddings model."""
    if settings.EMBEDDING_PROVIDER.lower() == "openai":
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY
        )
    else:
        # Fallback or extensible for other providers
        raise ValueError(f"Unsupported embedding provider: {settings.EMBEDDING_PROVIDER}")

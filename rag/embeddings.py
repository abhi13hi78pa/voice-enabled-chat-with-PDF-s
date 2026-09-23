import os
from langchain_openai import OpenAIEmbeddings
from config import settings


def get_embeddings_model():
    """
    Returns the configured embeddings model based on the environment configuration.
    Supports both OpenAI and NVIDIA embedding providers.
    Raises clear errors if required credentials or configurations are missing.
    """
    provider = settings.EMBEDDING_PROVIDER.lower()

    if provider == "openai":
        if (
            not settings.OPENAI_API_KEY
            or settings.OPENAI_API_KEY.strip() == ""
            or settings.OPENAI_API_KEY == "your_openai_api_key_here"
        ):
            raise ValueError(
                "OPENAI_API_KEY is missing or invalid. "
                "Please configure it in your .env file to use OpenAI embeddings."
            )

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
        )

    elif provider == "nvidia":
        if (
            not settings.NVIDIA_API_KEY
            or settings.NVIDIA_API_KEY.strip() == ""
            or settings.NVIDIA_API_KEY == "your_nvidia_api_key_here"
        ):
            raise ValueError(
                "NVIDIA_API_KEY is missing or invalid. "
                "Please configure it in your .env file to use NVIDIA embeddings."
            )

        # NVIDIA NIM provides an OpenAI-compatible embeddings endpoint.
        # We reuse langchain_openai.OpenAIEmbeddings pointed at the NVIDIA base URL.
        # IMPORTANT: check_embedding_ctx_length=False is required because otherwise
        # LangChain tokenizes the input into integer arrays, which NVIDIA NIMs reject.
        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            openai_api_key=settings.NVIDIA_API_KEY,
            openai_api_base=settings.NVIDIA_BASE_URL,
            check_embedding_ctx_length=False
        )

    else:
        raise ValueError(f"Unsupported embedding provider configured: {provider}")

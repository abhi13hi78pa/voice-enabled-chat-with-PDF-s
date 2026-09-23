import os
from dotenv import load_dotenv

load_dotenv()

# Project Config
PROJECT_NAME = os.getenv("PROJECT_NAME", "VoicePDF")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

# LLM Config (NVIDIA Nemotron default)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia")
LLM_MODEL = os.getenv("LLM_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embeddings Config
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:password@localhost:5432/voicepdf")

# Retrieval
TOP_K = int(os.getenv("TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

# Ingestion
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))

# Voice
STT_PROVIDER = os.getenv("STT_PROVIDER", "google")
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "google")

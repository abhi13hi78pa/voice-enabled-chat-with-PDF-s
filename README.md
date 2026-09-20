# VoicePDF: Voice-Enabled Conversational PDF Assistant

## Overview
VoicePDF is a production-quality, voice-enabled conversational RAG (Retrieval-Augmented Generation) application designed to interact intelligently with PDF documents. It incorporates concepts from the NVIDIA RAG course with a clean, extensible architecture.

**Current Status:** Phase 2 (PDF Ingestion & Chunking)

## Architecture
```mermaid
flowchart TD
    User([User])
    User -->|Voice/Text| UI[Gradio UI (Phase 2 Shell)]
    User -->|PDF Upload| Ingestion[Ingestion Service]
    
    subgraph Voice Services (Planned)
        UI -.->|Audio| STT[Speech-to-Text]
        TTS[Text-to-Speech] -.->|Audio| UI
    end
    
    subgraph Document Processing
        Ingestion --> Loader[PDF Loader]
        Loader --> Chunker[Chunker]
        Chunker -.-> Indexer[Indexer (Planned)]
    end
    
    subgraph RAG Pipeline (Planned)
        UI -.-> RAGChain[LangChain RAG Chain]
        STT -.-> RAGChain
        RAGChain -.-> Retriever[Vector Retriever]
        Retriever -.-> VectorDB[(PostgreSQL + pgvector)]
        RAGChain -.-> LLM[LLM / GenAI]
    end
    
    Indexer -.-> VectorDB
    LLM -.-> TTS
    LLM -.-> UI
```

## Tech Stack
- **Frontend**: Gradio
- **Backend/AI**: Python, LangChain
- **Database**: PostgreSQL with pgvector, SQLAlchemy
- **Embeddings/LLM**: Configurable via Environment Variables
- **Voice**: SpeechRecognition, gTTS (Planned)

## Current Implementation Status
- [x] Clean project architecture & directory structure. (Phase 1)
- [x] Configurable environment variables (`.env`). (Phase 1)
- [x] Database connection and SQLAlchemy models schema prepared. (Phase 1)
- [x] PostgreSQL + pgvector Docker setup. (Phase 1)
- [x] Basic UI shell built with Gradio. (Phase 1)
- [x] Foundational tests implemented. (Phase 1)
- [x] Complete PDF ingestion and chunking (Phase 2).
- [ ] Vector indexing and embeddings (Phase 3).
- [ ] Semantic retrieval (Phase 4).
- [ ] Complete RAG chain (Phase 5).
- [ ] Voice integration (Phase 9 & 10).

## Local Setup

1. **Python Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**:
   ```bash
   cp .env.example .env
   # Update variables as necessary
   ```

4. **PostgreSQL + pgvector Setup**:
   ```bash
   docker-compose up -d
   ```

5. **Run the Application (UI Shell)**:
   ```bash
   python app.py
   ```

6. **Run Tests**:
   ```bash
   pytest tests/
   ```

## Planned Future Phases
Phase 2 will implement the PDF ingestion and chunking pipeline. Phase 3-8 will handle Embeddings, Retrieval, LLM Integration, and Memory. Phases 9 and 10 will focus on Voice integration.

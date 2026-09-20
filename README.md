# VoicePDF: Voice-Enabled Conversational PDF Assistant

## Overview
VoicePDF is a production-quality, voice-enabled conversational RAG (Retrieval-Augmented Generation) application designed to interact intelligently with PDF documents. It incorporates concepts from the NVIDIA RAG course with a clean, extensible architecture.

**Current Status:** Phase 4 (LLM Generation & Grounded RAG)

## Architecture
```mermaid
flowchart TD
    User([User])
    User -->|Voice/Text| UI[Gradio UI (Phase 4 Shell)]
    User -->|PDF Upload| Ingestion[Ingestion Service]
    
    subgraph Voice Services (Planned)
        UI -.->|Audio| STT[Speech-to-Text]
        TTS[Text-to-Speech] -.->|Audio| UI
    end
    
    subgraph Document Processing
        Ingestion --> Loader[PDF Loader]
        Loader --> Chunker[Text Chunking]
        Chunker --> Embedder[Embedding Generation]
        Embedder --> Indexer[PostgreSQL + pgvector]
    end
    
    subgraph RAG Pipeline
        UI --> RAGChain[LangChain RAG Chain]
        STT -.-> RAGChain
        RAGChain --> Retriever[Vector Similarity Retrieval]
        Retriever --> VectorDB[(PostgreSQL + pgvector)]
        RAGChain --> Context[Context Builder]
        Context --> Prompt[Grounded Prompt]
        Prompt --> LLM[LLM / GenAI]
    end
    
    Indexer -.-> VectorDB
    LLM --> UI
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
- [x] Vector indexing and embeddings (Phase 3).
- [x] Semantic retrieval foundation (Phase 3).
- [x] Complete RAG chain & Grounded Generation (Phase 4).
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

## Design Decisions (Phase 3)
*   **Embeddings & Dimensions**: We utilize configurable embeddings (default `text-embedding-3-small` via OpenAI). The vector dimension in the pgvector database inherently matches this configuration (e.g., 1536).
*   **pgvector & Similarity Search**: Chunk embeddings are stored natively in PostgreSQL using the pgvector extension. This allows us to perform fast, accurate semantic similarity search (using vector distance) directly in our relational database without needing a separate vector DB microservice.
*   **Retriever Architecture**: The RAG Retriever queries pgvector with a given user prompt, computing the prompt's embedding on the fly and returning the `TOP_K` chunks.
*   **Metadata & Page Citation Preservation**: The `DocumentChunk` model stores a `metadata_json` field containing `document_name`, `page_number`, `chunk_index`, and a unique `chunk_id`. The retriever returns this payload entirely intact, allowing the future LLM generation phase to ground its answers with explicit, page-level citations.

## Design Decisions (Phase 4)
*   **Grounded RAG Generation**: The LLM prompt explicitly forces the model to use *only* the retrieved context and avoids hallucinating outside knowledge. If context is missing, it explicitly states it rather than guessing.
*   **Temperature Strategy**: The LLM temperature is set to `0.0` to ensure strict, deterministic answering based on facts.
*   **Citation Deduplication**: The context builder automatically parses the metadata from the semantic chunks and generates a deduplicated set of human-readable citations (e.g., `report.pdf — Page 3`), which are appended to the user response.

## Roadmap & Future Phases

### Completed Phases
- **Phase 1 — Foundation** ✅: Clean project structure, environment configuration, PostgreSQL + pgvector container, base UI shell.
- **Phase 2 — PDF Ingestion & Chunking** ✅: File validation, PyPDFLoader text extraction, 1-indexed page metadata, recursive chunking.
- **Phase 3 — Embeddings & pgvector Store** ✅: OpenAI embeddings, pgvector storage service, semantic retriever with metadata filtering.
- **Phase 4 — Grounded LLM RAG Generation** ✅: Strict prompt guardrails, context assembly, programmatic citation extraction, end-to-end RAG orchestrator.

### Upcoming Phases (Planned)
- **Phase 5 — Voice: Speech-to-Text & Text-to-Speech** (Planned): Microphone audio input (STT) and synthesized voice output (TTS).
- **Phase 6 — Conversation & Memory Improvements** (Planned): Multi-turn conversational memory and chat history context.
- **Phase 7 — Evaluation & Quality Improvements** (Planned): Retrieval and answering quality benchmarks.
- **Phase 8 — Production Polish & Deployment** (Planned): Full-stack containerization, health checks, and performance optimizations.

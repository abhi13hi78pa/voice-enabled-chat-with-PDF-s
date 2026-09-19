import datetime
from sqlalchemy import Column, String, Integer, DateTime, JSON, Text
from sqlalchemy.orm import declarative_base
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime, default=datetime.datetime.utcnow)
    status = Column(String, default='processing')

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    id = Column(String, primary_key=True)
    document_id = Column(String, nullable=False, index=True)
    page_number = Column(Integer)
    chunk_index = Column(Integer)
    content = Column(Text, nullable=False)
    # The vector dimension is dynamically matched or fixed in migration.
    # In pgvector 0.2+, dimension is optional if created dynamically.
    embedding = Column(Vector()) 
    metadata_json = Column(JSON)

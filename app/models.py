from sqlalchemy import Column, Integer, Text, TIMESTAMP
from sqlalchemy.sql import func
from app.database import Base


class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(Text, nullable=False)
    uploaded_by = Column(Text)
    upload_time = Column(TIMESTAMP, server_default=func.now())
    chunk_count = Column(Integer)
    embedding_model = Column(Text)


class ChatLog(Base):
    __tablename__ = "chat_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_query = Column(Text, nullable=False)
    ai_response = Column(Text, nullable=False)
    related_document = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
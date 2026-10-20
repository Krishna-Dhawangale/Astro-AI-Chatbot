import datetime
import uuid
from typing import Any, Dict, Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=True)
    email = Column(String(255), nullable=True, index=True)
    birth_date = Column(DateTime, nullable=True)
    birth_hour = Column(Integer, nullable=True)
    birth_place = Column(String(255), nullable=True)
    latitude = Column(Float, nullable=True, default=28.6139)
    longitude = Column(Float, nullable=True, default=77.2090)
    timezone = Column(String(50), nullable=True, default="5.5")
    lagna = Column(String(100), nullable=True)
    moon_sign = Column(String(100), nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )

    # Relationships
    sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "name": self.name,
            "email": self.email,
            "birth_date": self.birth_date.isoformat() if self.birth_date else None,
            "birth_hour": self.birth_hour,
            "birth_place": self.birth_place,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "lagna": self.lagna,
            "moon_sign": self.moon_sign,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(255), nullable=True)
    birth_profile = Column(JSONB, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
    )

    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "title": self.title,
            "birth_profile": self.birth_profile,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ChatMessage(Base):
    """
    ChatMessage model with failure tracking and error locking.
    Tracks all user queries, model responses, execution modes, and failure flags.
    Supported execution modes: 'LOCAL', 'HYBRID', 'LLM', 'UNRESOLVED', 'CRASH_FALLBACK'.
    """
    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=True, index=True)
    user_query = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    execution_mode = Column(String(50), nullable=False, default="LOCAL")  # LOCAL, HYBRID, LLM, UNRESOLVED, CRASH_FALLBACK
    is_unresolved = Column(Boolean, default=False, nullable=False, index=True)
    timestamp = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        default=lambda: datetime.datetime.now(datetime.timezone.utc),
        nullable=False,
        index=True,
    )

    # Telemetry and calculation metadata
    detected_intents = Column(JSONB, nullable=True)
    input_tokens = Column(Integer, default=0, nullable=False)
    output_tokens = Column(Integer, default=0, nullable=False)
    calculation_footer = Column(Text, nullable=True)

    # Relationships
    session = relationship("ChatSession", back_populates="messages")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id) if self.session_id else None,
            "user_query": self.user_query,
            "bot_response": self.bot_response,
            "execution_mode": self.execution_mode,
            "is_unresolved": self.is_unresolved,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "detected_intents": self.detected_intents,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calculation_footer": self.calculation_footer,
        }

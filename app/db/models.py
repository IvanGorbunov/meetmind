from sqlalchemy import String, Text, Integer
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from app.db.database import Base


class Transcript(Base):
    """Model for storing uploaded transcripts."""
    
    __tablename__ = "transcripts"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    uploaded_at: Mapped[int] = mapped_column(
        Integer,
        default=int(datetime.utcnow().timestamp()),
        nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<Transcript(id={self.id}, filename='{self.filename}')>"


class SearchHistory(Base):
    """Model for storing search history."""
    
    __tablename__ = "search_history"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    searched_at: Mapped[int] = mapped_column(
        Integer,
        default=int(datetime.utcnow().timestamp()),
        nullable=False
    )
    
    def __repr__(self) -> str:
        return f"<SearchHistory(id={self.id}, question='{self.question[:50]}...')>"

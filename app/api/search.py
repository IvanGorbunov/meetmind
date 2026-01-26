from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date, datetime, timedelta

from app.db.database import get_db
from app.db.models import SearchHistory
from app.services.rag import get_rag_service


router = APIRouter(prefix="/search", tags=["Search"])


class SearchRequest(BaseModel):
    """Request model for search."""
    question: str
    date_from: Optional[datetime] = Field(default=None, description="Start date for search (inclusive)")
    date_to: Optional[datetime] = Field(default=None, description="End date for search (inclusive)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "Когда обсуждали дедлайн проекта?",
                "date_from": "2024-01-01T12:00:00+03:00",
                "date_to": "2024-01-31T23:59:59+03:00"
            }
        }


class SourceInfo(BaseModel):
    """Source document info."""
    content: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response model for search."""
    question: str
    answer: str
    sources: List[SourceInfo]


class StatsResponse(BaseModel):
    """Response model for stats."""
    total_documents: int
    embeddings_provider: str
    llm_provider: str


@router.post("", response_model=SearchResponse)
async def search_transcripts(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Search for information in transcripts using RAG.
    
    Send a natural language question and get an AI-generated answer
    based on the indexed meeting transcripts.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    rag_service = get_rag_service()
    
    # Check if there are any documents
    stats = rag_service.get_stats()
    if stats["total_documents"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No transcripts indexed yet. Please upload some transcripts first."
        )
    
    # Calculate date range if not provided
    date_from = request.date_from
    date_to = request.date_to

    today = datetime.utcnow()
    if not date_from:
        date_from = today - timedelta(days=7)
    if not date_to:
        date_to = today
    
    # Perform search
    try:
        result = rag_service.search(
            question=request.question,
            date_from=date_from,
            date_to=date_to
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )
    
    # Save to search history
    history_entry = SearchHistory(
        question=request.question,
        answer=result["answer"]
    )
    db.add(history_entry)
    
    return SearchResponse(
        question=request.question,
        answer=result["answer"],
        sources=[
            SourceInfo(content=s["content"], metadata=s["metadata"])
            for s in result["sources"]
        ]
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get RAG system statistics.
    """
    from app.config import get_settings
    settings = get_settings()
    
    rag_service = get_rag_service()
    stats = rag_service.get_stats()
    
    return StatsResponse(
        total_documents=stats["total_documents"],
        embeddings_provider=settings.embeddings_provider,
        llm_provider=settings.llm_provider
    )

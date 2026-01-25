from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import List
from datetime import datetime

from app.db.database import get_db
from app.db.models import Transcript
from app.services.rag import get_rag_service


router = APIRouter(prefix="/transcripts", tags=["Transcripts"])


class TranscriptResponse(BaseModel):
    """Response model for transcript."""
    id: int
    filename: str
    uploaded_at: datetime
    chunks_indexed: int = 0
    
    class Config:
        from_attributes = True


class TranscriptListResponse(BaseModel):
    """Response model for transcript list."""
    items: List[TranscriptResponse]
    total: int


@router.post("", response_model=TranscriptResponse)
async def upload_transcript(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a transcript file.
    
    Accepts .txt files with meeting transcription content.
    The file will be saved to database and indexed for RAG search.
    """
    # Validate file type
    if not file.filename.endswith('.txt'):
        raise HTTPException(
            status_code=400,
            detail="Only .txt files are supported"
        )
    
    # Read file content
    try:
        content = await file.read()
        content_str = content.decode('utf-8')
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="File must be UTF-8 encoded"
        )
    
    if not content_str.strip():
        raise HTTPException(
            status_code=400,
            detail="File is empty"
        )
    
    # Save to database
    transcript = Transcript(
        filename=file.filename,
        content=content_str
    )
    db.add(transcript)
    await db.flush()
    await db.refresh(transcript)
    
    # Index in vector store
    rag_service = get_rag_service()
    chunks_count = rag_service.index_document(
        content=content_str,
        metadata={
            "transcript_id": transcript.id,
            "filename": transcript.filename,
            "uploaded_at": transcript.uploaded_at.isoformat()
        }
    )
    
    return TranscriptResponse(
        id=transcript.id,
        filename=transcript.filename,
        uploaded_at=transcript.uploaded_at,
        chunks_indexed=chunks_count
    )


@router.get("", response_model=TranscriptListResponse)
async def list_transcripts(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
):
    """
    List all uploaded transcripts.
    
    Returns paginated list of transcripts sorted by upload date (newest first).
    """
    # Get total count
    count_result = await db.execute(select(Transcript))
    total = len(count_result.scalars().all())
    
    # Get paginated items
    result = await db.execute(
        select(Transcript)
        .order_by(Transcript.uploaded_at.desc())
        .offset(skip)
        .limit(limit)
    )
    items = result.scalars().all()
    
    return TranscriptListResponse(
        items=[
            TranscriptResponse(
                id=t.id,
                filename=t.filename,
                uploaded_at=t.uploaded_at
            )
            for t in items
        ],
        total=total
    )


@router.get("/{transcript_id}")
async def get_transcript(
    transcript_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific transcript by ID.
    """
    result = await db.execute(
        select(Transcript).where(Transcript.id == transcript_id)
    )
    transcript = result.scalar_one_or_none()
    
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found")
    
    return {
        "id": transcript.id,
        "filename": transcript.filename,
        "content": transcript.content,
        "uploaded_at": transcript.uploaded_at
    }

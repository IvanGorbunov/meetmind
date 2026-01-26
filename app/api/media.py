"""
Media upload and transcription API endpoints.
"""
import os
import uuid
import logging
from pathlib import Path

import aiofiles
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime

from app.db.database import get_db
from app.db.models import Transcript
from app.config import get_settings
from app.services.transcription import get_transcription_service
from app.services.rag import get_rag_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["Media"])


class TranscribeResponse(BaseModel):
    """Response model for transcription."""
    id: int
    filename: str
    content: str
    uploaded_at: datetime
    chunks_indexed: int = 0
    
    class Config:
        from_attributes = True


ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.webm', '.ogg', '.flac'}


def validate_audio_file(filename: str) -> bool:
    """Validate that file has allowed audio extension."""
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


@router.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_media(
    file: UploadFile = File(...),
    language: str = "ru",
    db: AsyncSession = Depends(get_db)
):
    """
    Upload and transcribe an audio file.
    
    Accepts audio files (mp3, wav, m4a, webm, ogg, flac) and returns
    the transcribed text. The transcription is saved to database
    and indexed for RAG search.
    
    Args:
        file: Audio file to transcribe
        language: Language code (default: "ru" for Russian)
    
    Returns:
        TranscribeResponse with transcript ID, content, and indexing stats
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")
    
    if not validate_audio_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    
    settings = get_settings()
    
    # Ensure upload directory exists
    upload_dir = Path(settings.media_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    ext = Path(file.filename).suffix.lower()
    unique_filename = f"{uuid.uuid4()}{ext}"
    file_path = upload_dir / unique_filename
    
    try:
        # Save uploaded file
        async with aiofiles.open(file_path, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        logger.info(f"Saved audio file: {file_path}")
        
        # Transcribe
        transcription_service = get_transcription_service()
        transcript_text = transcription_service.transcribe(
            str(file_path),
            language=language
        )
        
        if not transcript_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Transcription resulted in empty text"
            )
        
        # Save to database
        transcript = Transcript(
            filename=file.filename,
            content=transcript_text
        )
        db.add(transcript)
        await db.flush()
        await db.refresh(transcript)
        
        # Index in vector store
        rag_service = get_rag_service()
        chunks_count = rag_service.index_document(
            content=transcript_text,
            metadata={
                "transcript_id": transcript.id,
                "filename": transcript.filename,
                "uploaded_at": transcript.uploaded_at.isoformat(),
                "source_type": "audio"
            }
        )
        
        logger.info(f"Transcription complete. ID: {transcript.id}, chunks: {chunks_count}")
        
        return TranscribeResponse(
            id=transcript.id,
            filename=transcript.filename,
            content=transcript_text,
            uploaded_at=transcript.uploaded_at,
            chunks_indexed=chunks_count
        )
        
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise HTTPException(status_code=500, detail="Failed to process audio file")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Cleanup temporary file
        if file_path.exists():
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")

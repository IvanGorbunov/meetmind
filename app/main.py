from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.db.database import init_db
from app.api.transcripts import router as transcripts_router
from app.api.search import router as search_router
from app.api.media import router as media_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    await init_db()
    yield
    # Shutdown
    pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="MeetMind",
        description="RAG-based Meeting Transcript Search API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(transcripts_router)
    app.include_router(search_router)
    app.include_router(media_router)
    
    @app.get("/", tags=["Health"])
    async def root():
        """Health check endpoint."""
        return {
            "status": "ok",
            "service": "MeetMind",
            "version": "1.0.0",
            "providers": {
                "embeddings": settings.embeddings_provider,
                "llm": settings.llm_provider
            }
        }
    
    @app.get("/health", tags=["Health"])
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}
    
    return app


app = create_app()

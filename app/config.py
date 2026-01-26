from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Provider selection
    embeddings_provider: Literal["openai", "local", "huggingface"] = Field(
        default="local",
        description="Embeddings provider: openai, local, or huggingface"
    )
    llm_provider: Literal["openai", "local", "huggingface"] = Field(
        default="local", 
        description="LLM provider: openai, local, or huggingface"
    )
    
    # OpenAI
    openai_api_key: str = Field(default="", description="OpenAI API key")
    
    # HuggingFace
    huggingface_api_token: str = Field(default="", description="HuggingFace API token")
    
    # Local embedding model
    local_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace model for local embeddings"
    )
    
    # Ollama
    ollama_model: str = Field(default="llama3", description="Ollama model name")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    
    # HuggingFace models
    hf_embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace embedding model ID"
    )
    hf_llm_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="HuggingFace LLM model ID"
    )
    
    # Database
    database_url: str = Field(
        default="sqlite+aiosqlite:///./meetmind.db",
        description="Database connection URL"
    )
    
    # ChromaDB
    chroma_persist_directory: str = Field(
        default="./chroma_db",
        description="ChromaDB persistence directory"
    )
    
    # WhisperX (Audio Transcription)
    whisperx_model: str = Field(
        default="large-v3",
        description="WhisperX model size: tiny, base, small, medium, large-v2, large-v3"
    )
    whisperx_device: str = Field(
        default="cuda",
        description="Device for WhisperX: cuda or cpu"
    )
    whisperx_compute_type: str = Field(
        default="float16",
        description="Compute type: float16, int8 (for GPU), or float32 (for CPU)"
    )
    media_upload_dir: str = Field(
        default="./media_uploads",
        description="Directory for temporary media file uploads"
    )
    
    # RAG Configuration
    rag_prompt_template: str = Field(
        default="""Ты ассистент для анализа рабочих созвонов.
Отвечай только на основе предоставленного контекста.
Если информации нет в контексте, скажи об этом честно.

Контекст:
{context}

Вопрос: {question}

Ответ:""",
        description="RAG prompt template with {context} and {question} placeholders"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

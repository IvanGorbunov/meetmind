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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

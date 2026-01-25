from abc import ABC, abstractmethod
from typing import List

from langchain_core.embeddings import Embeddings

from app.config import get_settings


class BaseEmbeddings(ABC):
    """Abstract base class for embeddings providers."""
    
    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return LangChain Embeddings instance."""
        pass


def get_embeddings() -> Embeddings:
    """
    Factory function to get embeddings based on configuration.
    Returns the appropriate embeddings instance based on EMBEDDINGS_PROVIDER env var.
    """
    settings = get_settings()
    provider = settings.embeddings_provider
    
    if provider == "openai":
        from app.services.embeddings.openai import OpenAIEmbeddingsProvider
        return OpenAIEmbeddingsProvider().get_embeddings()
    
    elif provider == "local":
        from app.services.embeddings.local import LocalEmbeddingsProvider
        return LocalEmbeddingsProvider().get_embeddings()
    
    elif provider == "huggingface":
        from app.services.embeddings.huggingface import HuggingFaceEmbeddingsProvider
        return HuggingFaceEmbeddingsProvider().get_embeddings()
    
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")

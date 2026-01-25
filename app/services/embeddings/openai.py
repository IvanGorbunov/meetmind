from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.services.embeddings.base import BaseEmbeddings


class OpenAIEmbeddingsProvider(BaseEmbeddings):
    """OpenAI embeddings provider using text-embedding-3-small model."""
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_embeddings(self) -> Embeddings:
        """Return OpenAI Embeddings instance."""
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings provider")
        
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.settings.openai_api_key
        )

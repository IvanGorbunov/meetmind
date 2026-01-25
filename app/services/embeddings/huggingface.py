from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.services.embeddings.base import BaseEmbeddings


class HuggingFaceEmbeddingsProvider(BaseEmbeddings):
    """
    HuggingFace Inference API embeddings provider.
    Uses HuggingFace API for embedding generation (requires API token).
    """
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_embeddings(self) -> Embeddings:
        """Return HuggingFace Inference API Embeddings instance."""
        if not self.settings.huggingface_api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN is required for HuggingFace embeddings provider")
        
        return HuggingFaceEndpointEmbeddings(
            model=self.settings.hf_embedding_model,
            huggingfacehub_api_token=self.settings.huggingface_api_token
        )

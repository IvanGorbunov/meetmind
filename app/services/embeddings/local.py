from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from app.config import get_settings
from app.services.embeddings.base import BaseEmbeddings


class LocalEmbeddingsProvider(BaseEmbeddings):
    """
    Local embeddings provider using HuggingFace models.
    Runs on local GPU/CPU without API calls.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_embeddings(self) -> Embeddings:
        """Return local HuggingFace Embeddings instance."""
        import torch
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        return HuggingFaceEmbeddings(
            model_name=self.settings.local_embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )

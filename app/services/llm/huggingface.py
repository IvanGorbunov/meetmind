from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.language_models import BaseChatModel

from app.config import get_settings
from app.services.llm.base import BaseLLM


class HuggingFaceLLMProvider(BaseLLM):
    """
    HuggingFace Inference API LLM provider.
    Uses HuggingFace API for text generation (requires API token).
    """
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_llm(self) -> BaseChatModel:
        """Return HuggingFace Inference API LLM instance."""
        if not self.settings.huggingface_api_token:
            raise ValueError("HUGGINGFACE_API_TOKEN is required for HuggingFace LLM provider")
        
        return HuggingFaceEndpoint(
            repo_id=self.settings.hf_llm_model,
            huggingfacehub_api_token=self.settings.huggingface_api_token,
            temperature=0.1,
            max_new_tokens=512
        )

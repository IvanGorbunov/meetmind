from langchain_ollama import ChatOllama
from langchain_core.language_models import BaseChatModel

from app.config import get_settings
from app.services.llm.base import BaseLLM


class LocalLLMProvider(BaseLLM):
    """
    Local LLM provider using Ollama.
    Requires Ollama to be running locally.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_llm(self) -> BaseChatModel:
        """Return Ollama Chat Model instance."""
        return ChatOllama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=0
        )

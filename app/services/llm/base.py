from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from app.config import get_settings


class BaseLLM(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        """Return LangChain Chat Model instance."""
        pass


def get_llm() -> BaseChatModel:
    """
    Factory function to get LLM based on configuration.
    Returns the appropriate LLM instance based on LLM_PROVIDER env var.
    """
    settings = get_settings()
    provider = settings.llm_provider
    
    if provider == "openai":
        from app.services.llm.openai import OpenAILLMProvider
        return OpenAILLMProvider().get_llm()
    
    elif provider == "local":
        from app.services.llm.local import LocalLLMProvider
        return LocalLLMProvider().get_llm()
    
    elif provider == "huggingface":
        from app.services.llm.huggingface import HuggingFaceLLMProvider
        return HuggingFaceLLMProvider().get_llm()
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

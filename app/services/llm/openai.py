from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.config import get_settings
from app.services.llm.base import BaseLLM


class OpenAILLMProvider(BaseLLM):
    """OpenAI LLM provider using GPT-4 Turbo."""
    
    def __init__(self):
        self.settings = get_settings()
        
    def get_llm(self) -> BaseChatModel:
        """Return OpenAI Chat Model instance."""
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI LLM provider")
        
        return ChatOpenAI(
            model="gpt-4-turbo-preview",
            temperature=0,
            openai_api_key=self.settings.openai_api_key
        )

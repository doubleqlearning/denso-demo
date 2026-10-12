from abc import ABC, abstractmethod
from typing import List, Generator
from backend.utils.schema import Message

class BaseLLM(ABC):
    """
    Abstract Base Class for all LLM providers.
    Defines a common interface for chat, streaming, and tool-augmented conversations.
    """

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def chat(self, messages: List[Message]) -> str:
        """Perform a standard synchronous chat completion."""
        pass

    @abstractmethod
    def chat_streaming(self, messages: List[Message]) -> Generator[str, None, None]:
        """Perform a streaming chat completion."""
        pass

    @abstractmethod
    def chat_with_tools(self, messages: List[Message]) -> Generator[str, None, None]:
        """Perform a tool-augmented chat. Handles calling tools and sending follow-up messages."""
        pass

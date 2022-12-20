"""Base interface for logging runs."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLogger(ABC):
    """Base interface for logging runs."""

    @abstractmethod
    def log_llm_run_start(self, serialized: Dict[str, Any], prompts: List[str], **extra: str) -> None:
        """Log the start of an LLM run."""

    @abstractmethod
    def log_llm_run_end(self, response: Dict[str, Any], error=None) -> None:
        """Log the end of an LLM run."""

    @abstractmethod
    def log_chain_run_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str) -> None:
        """Log the start of a chain run."""

    @abstractmethod
    def log_chain_run_end(self, outputs: Dict[str, Any], error=None) -> None:
        """Log the end of a chain run."""

    @abstractmethod
    def log_tool_run_start(self, serialized: Dict[str, Any], action: str, inputs: Dict[str, Any], **extra: str) -> None:
        """Log the start of a tool run."""

    @abstractmethod
    def log_tool_run_end(self, outputs: Dict[str, Any], error=None) -> None:
        """Log the end of a tool run."""

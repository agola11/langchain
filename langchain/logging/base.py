"""Base interface for logging runs."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Run:
    id: int
    start_time: datetime
    end_time: datetime
    extra: Dict[str, Any]
    error: Dict[str, Any]
    execution_order: int
    serialized: Dict[str, Any]


@dataclass
class LLMRun(Run):
    prompts: Dict[str, Any]
    response: Dict[str, Any]


@dataclass
class ChainRun(Run):
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    child_llm_runs: List[Run]
    child_chain_runs: List[Run]
    child_tool_runs: List[Run]


@dataclass
class ToolRun(Run):
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    action: str
    child_llm_runs: List[Run]
    child_chain_runs: List[Run]
    child_tool_runs: List[Run]


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

    @abstractmethod
    def get_llm_runs(self) -> List[LLMRun]:
        """Get all LLM runs."""

    @abstractmethod
    def get_chain_runs(self) -> List[ChainRun]:
        """Get all chain runs."""

    @abstractmethod
    def get_tool_runs(self) -> List[ToolRun]:
        """Get all tool runs."""

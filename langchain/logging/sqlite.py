import datetime
import threading
from typing import Any, Dict, List, Tuple, Type, Union

from sqlalchemy import and_, select

from langchain.logging import base
from langchain.logging.base import BaseLogger
from langchain.logging.models.models import ChainRun, LLMRun, ToolRun, session
from dataclasses import dataclass, field


class LoggerException(Exception):
    """Base class for exceptions in logging module."""


def _deep_convert_run(
    run: Union[LLMRun, ChainRun, ToolRun]
) -> Union[base.LLMRun, base.ChainRun, base.ToolRun]:
    """Converts a run to a base run."""

    if isinstance(run, LLMRun):
        return base.LLMRun(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            extra=run.extra,
            error=run.error,
            execution_order=run.execution_order,
            serialized=run.serialized,
            prompts=run.prompts,
            response=run.response,
        )

    # Get all the nested runs of a run.
    child_llm_runs = [_deep_convert_run(llm_run) for llm_run in run.child_llm_runs]
    child_chain_runs = run.child_chain_runs
    child_tool_runs = run.child_tool_runs
    nested_chain_runs = [_deep_convert_run(cr) for cr in child_chain_runs]
    nested_tool_runs = [_deep_convert_run(tr) for tr in child_tool_runs]

    child_runs = child_llm_runs + nested_chain_runs + nested_tool_runs
    child_runs.sort(key=lambda x: x.execution_order)
    if isinstance(run, ChainRun):
        return base.ChainRun(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            extra=run.extra,
            error=run.error,
            execution_order=run.execution_order,
            serialized=run.serialized,
            inputs=run.inputs,
            outputs=run.outputs,
            child_runs=child_runs,
        )
    elif isinstance(run, ToolRun):
        return base.ToolRun(
            id=run.id,
            start_time=run.start_time,
            end_time=run.end_time,
            extra=run.extra,
            error=run.error,
            execution_order=run.execution_order,
            serialized=run.serialized,
            inputs=run.inputs,
            outputs=run.outputs,
            action=run.action,
            child_runs=child_runs,
        )


def _get_runs(
    run_type: Type[Union[LLMRun, ChainRun, ToolRun]], top_level_only: bool
) -> List[Union[base.LLMRun, base.ChainRun, base.ToolRun]]:
    """Get all runs of a given type."""

    if top_level_only:
        return [
            _deep_convert_run(run)
            for run in session.scalars(
                select(run_type).where(
                    and_(
                        run_type.parent_chain_run == None,
                        run_type.parent_tool_run == None,
                    )
                )
            ).all()
        ]
    return [_deep_convert_run(run) for run in session.scalars(select(run_type)).all()]


def _get_run(
    run_type: Type[Union[LLMRun, ChainRun, ToolRun]], run_id: int
) -> Union[base.LLMRun, base.ChainRun, base.ToolRun]:
    """Get a specific run of a given type."""

    run = session.scalars(select(run_type).where(run_type.id == run_id)).first()
    if run is None:
        raise LoggerException(f"No {run_type.__name__} found with id {run_id}")
    return _deep_convert_run(run)


@dataclass
class LoggerStack(threading.local):
    """A stack of runs used for logging."""

    stack: List[Union[LLMRun, ChainRun, ToolRun]] = field(default_factory=list)
    execution_order: int = 1


class SqliteLogger(BaseLogger):
    """A logger that stores the logs in a sqlite database."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SqliteLogger, cls).__new__(cls)
            cls._instance._logger_stack = LoggerStack()
        return cls._instance

    def _log_run_start(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Log the start of a run."""

        run.execution_order = self._logger_stack.execution_order
        self._logger_stack.execution_order += 1

        if self._logger_stack.stack:
            if not (
                    isinstance(self._logger_stack.stack[-1], ChainRun)
                    or isinstance(self._logger_stack.stack[-1], ToolRun)
            ):
                session.rollback()
                raise LoggerException(
                    f"Nested {run.__class__.__name__} can only be logged inside a ChainRun or ToolRun"
                )
            if isinstance(run, LLMRun):
                self._logger_stack.stack[-1].child_llm_runs.append(run)
            elif isinstance(run, ChainRun):
                self._logger_stack.stack[-1].child_chain_runs.append(run)
            else:
                self._logger_stack.stack[-1].child_tool_runs.append(run)
        self._logger_stack.stack.append(run)
        run.save()

    def _end_log_run(self) -> None:
        """Call at the end of a run."""

        if not self._logger_stack.stack:
            session.commit()
            self._logger_stack.execution_order = 1

    def log_llm_run_start(
        self, serialized: Dict[str, Any], prompts: List[str], **extra: str
    ) -> None:
        """Log the start of an LLM run."""

        llm_run = LLMRun(
            serialized=serialized,
            prompts={"prompts": prompts},
            extra=extra,
            start_time=datetime.datetime.utcnow(),
        )
        self._log_run_start(llm_run)

    def log_llm_run_end(self, response: Dict[str, Any], error=None) -> None:
        """Log the end of an LLM run."""

        if not self._logger_stack.stack:
            raise LoggerException("No LLMRun found to be logged")

        llm_run = self._logger_stack.stack.pop()
        if not isinstance(llm_run, LLMRun):
            session.rollback()
            raise LoggerException("LLMRun end can only be logged after a LLMRun start")

        llm_run.update(response=response, error=error, end_time=datetime.datetime.utcnow())
        self._end_log_run()

    def log_chain_run_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str
    ) -> None:
        """Log the start of a chain run."""

        chain_run = ChainRun(
            serialized=serialized,
            inputs=inputs,
            extra=extra,
            start_time=datetime.datetime.utcnow(),
        )
        self._log_run_start(chain_run)

    def log_chain_run_end(self, outputs: Dict[str, Any], error=None) -> None:
        """Log the end of a chain run."""

        if not self._logger_stack.stack:
            raise LoggerException("No ChainRun found to be logged")

        chain_run = self._logger_stack.stack.pop()
        if not isinstance(chain_run, ChainRun):
            session.rollback()
            raise LoggerException(
                "ChainRun end can only be logged after a ChainRun start"
            )

        chain_run.update(outputs=outputs, error=error, end_time=datetime.datetime.utcnow())
        self._end_log_run()

    def log_tool_run_start(
        self,
        serialized: Dict[str, Any],
        action: str,
        inputs: Dict[str, Any],
        **extra: str,
    ) -> None:
        """Log the start of a tool run."""

        tool_run = ToolRun(
            serialized=serialized,
            action=action,
            inputs=inputs,
            extra=extra,
            start_time=datetime.datetime.utcnow(),
        )
        self._log_run_start(tool_run)

    def log_tool_run_end(self, outputs: Dict[str, Any], error=None) -> None:
        """Log the end of a tool run."""

        if not self._logger_stack.stack:
            raise LoggerException("No ToolRun found to be logged")

        tool_run = self._logger_stack.stack.pop()
        if not isinstance(tool_run, ToolRun):
            session.rollback()
            raise LoggerException(
                "ToolRun end can only be logged after a ToolRun start"
            )
        tool_run.update(outputs=outputs, error=error, end_time=datetime.datetime.utcnow())
        self._end_log_run()

    def get_llm_runs(self, top_level_only: bool = False) -> List[base.LLMRun]:
        """Return all the LLM runs."""

        return _get_runs(LLMRun, top_level_only)

    # TODO: specify nesting, utilize joined loads
    def get_chain_runs(self, top_level_only: bool = False) -> List[base.ChainRun]:
        """Return all the chain runs."""

        return _get_runs(ChainRun, top_level_only)

    # TODO: specify nesting, utilize joined loads
    def get_tool_runs(self, top_level_only: bool = False) -> List[base.ToolRun]:
        """Return all the tool runs."""

        return _get_runs(ToolRun, top_level_only)

    def get_llm_run(self, run_id: int) -> base.LLMRun:
        """Return a specific LLM run."""

        return _get_run(LLMRun, run_id)

    def get_chain_run(self, run_id: int) -> base.ChainRun:
        """Return a specific chain run."""

        return _get_run(ChainRun, run_id)

    def get_tool_run(self, run_id: int) -> base.ToolRun:
        """Return a specific tool run."""

        return _get_run(ToolRun, run_id)

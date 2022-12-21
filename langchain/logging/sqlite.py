from langchain.logging.base import BaseLogger
from sqlalchemy import Column, ForeignKey, Integer, Table, DateTime, String, Boolean
from sqlalchemy.orm import declarative_base, relationship
from typing import Union, Any, Dict, List, Tuple

from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_mixin
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from langchain.logging import base
from sqlalchemy import select

Base = declarative_base()


# Assumptions:
# 1. A chain run can have multiple LLM runs, chain runs, and tool runs
# 2. A tool run can have multiple LLM runs, chain runs, and tool runs
# 3. An LLM cannot have any child runs

@declarative_mixin
class RunMixin:
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, default=datetime.datetime.utcnow)
    extra = Column(JSON, default=None)
    error = Column(JSON, default=None)
    execution_order = Column(Integer, default=1)
    serialized = Column(JSON)


class LLMRun(Base, RunMixin):
    __tablename__ = "llm_runs"

    id = Column(Integer, primary_key=True)
    prompts = Column(JSON)
    response = Column(JSON)

    chain_run_id = Column(Integer, ForeignKey("chain_runs.id"))
    chain_run = relationship("ChainRun", back_populates="child_llm_runs")

    tool_run_id = Column(Integer, ForeignKey("tool_runs.id"))
    tool_run = relationship("ToolRun", back_populates="child_llm_runs")

    def __repr__(self):
        return f"<LLMRun(serialized={self.serialized}, execution_order={self.execution_order}, id={self.id}, prompts={self.prompts}, response={self.response})>"


ct_association_table = Table(
    "chain_runs_tool_runs",
    Base.metadata,
    Column("parent_chain_run_id", ForeignKey("chain_runs.id")),
    Column("child_tool_run_id", ForeignKey("tool_runs.id")),
)

tc_association_table = Table(
    "tool_runs_chain_runs",
    Base.metadata,
    Column("parent_tool_run_id", ForeignKey("tool_runs.id")),
    Column("child_chain_run_id", ForeignKey("chain_runs.id")),
)


class ChainRun(Base, RunMixin):
    __tablename__ = "chain_runs"

    id = Column(Integer, primary_key=True)
    inputs = Column(JSON)
    outputs = Column(JSON)

    child_llm_runs = relationship("LLMRun", back_populates="chain_run")

    parent_chain_run_id = Column(Integer, ForeignKey("chain_runs.id"))
    parent_chain_run = relationship("ChainRun", remote_side=[id], backref="child_chain_runs")

    child_tool_runs = relationship("ToolRun", secondary=ct_association_table, back_populates="parent_chain_run")
    parent_tool_run = relationship("ToolRun", secondary=tc_association_table, back_populates="child_chain_runs")

    def __repr__(self):
        return f"<ChainRun(serialized={self.serialized}, execution_order={self.execution_order}, id={self.id}, inputs={self.inputs}, outputs={self.outputs})>"


class ToolRun(Base, RunMixin):
    __tablename__ = "tool_runs"

    id = Column(Integer, primary_key=True)
    action = Column(String)
    inputs = Column(JSON)
    outputs = Column(JSON)

    child_llm_runs = relationship("LLMRun", back_populates="tool_run")

    parent_tool_run_id = Column(Integer, ForeignKey("tool_runs.id"))
    parent_tool_run = relationship("ToolRun", remote_side=[id], backref="child_tool_runs")

    child_chain_runs = relationship("ChainRun", secondary=tc_association_table, back_populates="parent_tool_run")
    parent_chain_run = relationship("ChainRun", secondary=ct_association_table, back_populates="child_tool_runs")

    def __repr__(self):
        return f"<ToolRun(serialized={self.serialized}, execution_order={self.execution_order}, inputs={self.inputs}, outputs={self.outputs})>"


def main() -> None:
    # engine = create_engine("sqlite:///log.db", echo=True, future=True) # Persistent
    engine = create_engine("sqlite://", echo=False, future=True)  # In-Memory
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        zeroshot_run = ChainRun(
            inputs={"text": "ZeroShot input"},
            outputs={"response": "ZeroShot output"},
            serialized={"name": "ZeroShot"},
            execution_order=15,
            child_chain_runs=[
                ChainRun(
                    inputs={"text": "LLMChain input1"},
                    outputs={"response": "LLMChain output1"},
                    serialized={"name": "LLMChain"},
                    execution_order=2,
                    child_llm_runs=[
                        LLMRun(
                            prompts={"text": "LLM input1"},
                            response={"response": "LLM output1"},
                            serialized={"name": "OpenAI"},
                            execution_order=1
                        )
                    ]
                ),
                ChainRun(
                    inputs={"text": "LLMChain input2"},
                    outputs={"response": "LLMChain output2"},
                    serialized={"name": "LLMChain"},
                    execution_order=5,
                    child_llm_runs=[
                        LLMRun(
                            prompts={"text": "LLM input2"},
                            response={"response": "LLM output2"},
                            serialized={"name": "OpenAI"},
                            execution_order=4,
                        )
                    ]
                ),
                ChainRun(
                    inputs={"text": "LLMChain input3"},
                    outputs={"response": "LLMChain output3"},
                    serialized={"name": "LLMChain"},
                    execution_order=8,
                    child_llm_runs=[
                        LLMRun(
                            prompts={"text": "LLM input3"},
                            response={"response": "LLM output3"},
                            serialized={"name": "OpenAI"},
                            execution_order=7,
                        )
                    ]
                ),
                ChainRun(
                    inputs={"text": "LLMChain input4"},
                    outputs={"response": "LLMChain output4"},
                    serialized={"name": "LLMChain"},
                    execution_order=14,
                    child_llm_runs=[
                        LLMRun(
                            prompts={"text": "LLM input4"},
                            response={"response": "LLM output4"},
                            serialized={"name": "OpenAI"},
                            execution_order=13,
                        )
                    ]
                )
            ],
            child_tool_runs=[
                ToolRun(
                    action="SEARCH",
                    inputs={"text": "Search input1"},
                    outputs={"response": "Search output1"},
                    serialized={"name": "SerpAPIWrapper"},
                    execution_order=3,
                ),
                ToolRun(
                    action="SEARCH",
                    inputs={"text": "Search input2"},
                    outputs={"response": "Search output2"},
                    serialized={"name": "SerpAPIWrapper"},
                    execution_order=6,
                ),
                ToolRun(
                    action="CALCULATOR",
                    inputs={"text": "Calculator input1"},
                    outputs={"response": "Calculator output1"},
                    serialized={"name": "LLMMathChain"},
                    execution_order=12,
                    child_chain_runs=[
                        ChainRun(
                            inputs={"text": "LLMMathChain input1"},
                            outputs={"response": "LLMMathChain output1"},
                            serialized={"name": "LLMMathChain"},
                            execution_order=11,
                            child_chain_runs=[
                                ChainRun(
                                    inputs={"text": "LLMChain input1"},
                                    outputs={"response": "LLMChain output1"},
                                    serialized={"name": "LLMChain"},
                                    execution_order=10,
                                    child_llm_runs=[
                                        LLMRun(
                                            prompts={"text": "LLM input1"},
                                            response={"response": "LLM output1"},
                                            serialized={"name": "OpenAI"},
                                            execution_order=9
                                        )
                                    ]
                                )
                            ]
                        )
                    ]
                )
            ]
        )
        session.add(zeroshot_run)
        session.commit()

    from sqlalchemy.orm import joinedload

    stmt = select(ChainRun).where(ChainRun.id == 1).options(joinedload(ChainRun.child_llm_runs),
                                                            joinedload(ChainRun.child_chain_runs).joinedload(
                                                                ChainRun.child_llm_runs),
                                                            joinedload(ChainRun.child_tool_runs))
    zeroshot_chain = session.scalars(stmt).unique().one()
    print_run(zeroshot_chain, "")


def print_run(run: Union[LLMRun, ChainRun, ToolRun], tabs: str) -> None:
    print(f"{tabs}Run: {run}")
    if isinstance(run, ChainRun) or isinstance(run, ToolRun):
        all_runs = run.child_chain_runs + run.child_llm_runs + run.child_tool_runs
        all_runs.sort(key=lambda x: x.execution_order)
        for child_run in all_runs:
            print_run(child_run, tabs + "\t")


def print_base_run(run: base.Run, tabs: str) -> None:
    print(f"{tabs}Run: {run}")
    if isinstance(run, base.ChainRun) or isinstance(run, base.ToolRun):
        all_runs = run.child_chain_runs + run.child_llm_runs + run.child_tool_runs
        all_runs.sort(key=lambda x: x.execution_order)
        for child_run in all_runs:
            print_base_run(child_run, tabs + "\t")


if __name__ == "__main__":
    main()


class LoggerException(Exception):
    """Base class for exceptions in logging module."""


def _deep_convert_run(run) -> Union[base.ChainRun, base.ToolRun]:
    """Get all the nested runs of a run."""

    # Get all the nested runs of a run.
    child_llm_runs = [_convert_llm_run(llm_run) for llm_run in run.child_llm_runs]
    child_chain_runs = run.child_chain_runs
    child_tool_runs = run.child_tool_runs
    nested_chain_runs = [_deep_convert_run(cr) for cr in child_chain_runs]
    nested_tool_runs = [_deep_convert_run(tr) for tr in child_tool_runs]
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
            child_llm_runs=child_llm_runs,
            child_chain_runs=nested_chain_runs,
            child_tool_runs=nested_tool_runs
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
            child_llm_runs=child_llm_runs,
            child_chain_runs=nested_chain_runs,
            child_tool_runs=nested_tool_runs
        )


def _convert_llm_run(llm_run) -> base.LLMRun:
    """Convert an LLMRun to a base.LLMRun."""

    return base.LLMRun(
            id=llm_run.id,
            start_time=llm_run.start_time,
            end_time=llm_run.end_time,
            extra=llm_run.extra,
            error=llm_run.error,
            execution_order=llm_run.execution_order,
            serialized=llm_run.serialized,
            prompts=llm_run.prompts,
            response=llm_run.response,
        )


class SqliteLogger(BaseLogger):
    """A logger that stores the logs in a sqlite database."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SqliteLogger, cls).__new__(cls)
            # TODO: make the db initialization string an environment variable
            cls._instance._db = create_engine("sqlite://", echo=False, future=True)
            Base.metadata.create_all(cls._instance._db)
            cls._instance._stack = []
            cls._instance._session = Session(cls._instance._db)
            cls._instance._execution_order = 1
        return cls._instance

    def _log_run_start(self, run: Union[LLMRun, ChainRun, ToolRun]) -> None:
        """Log the start of a run."""

        if self._stack:
            if not (isinstance(self._stack[-1], ChainRun) or isinstance(self._stack[-1], ToolRun)):
                self._session.rollback()
                raise LoggerException(f"Nested {run.__class__.__name__} can only be logged inside a ChainRun or ToolRun")
            if isinstance(run, LLMRun):
                self._stack[-1].child_llm_runs.append(run)
            elif isinstance(run, ChainRun):
                self._stack[-1].child_chain_runs.append(run)
            else:
                self._stack[-1].child_tool_runs.append(run)
        self._stack.append(run)
        self._session.add(run)

    def _end_log_run(self) -> None:
        """Call at the end of a run."""

        self._execution_order += 1
        if not self._stack:
            self._session.commit()
            self._execution_order = 1

    def log_llm_run_start(self, serialized: Dict[str, Any], prompts: List[str], **extra: str) -> None:
        """Log the start of an LLM run."""

        llm_run = LLMRun(serialized=serialized, prompts={"prompts": prompts}, extra=extra)
        self._log_run_start(llm_run)

    def log_llm_run_end(self, response: Dict[str, Any], error=None) -> None:
        """Log the end of an LLM run."""

        if not self._stack:
            raise LoggerException("No LLMRun found to be logged")

        llm_run = self._stack.pop()
        if not isinstance(llm_run, LLMRun):
            self._session.rollback()
            raise LoggerException("LLMRun end can only be logged after a LLMRun start")
        llm_run.response = response
        llm_run.error = error
        llm_run.end_time = datetime.datetime.utcnow()
        llm_run.execution_order = self._execution_order
        self._end_log_run()

    def log_chain_run_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **extra: str) -> None:
        """Log the start of a chain run."""

        chain_run = ChainRun(serialized=serialized, inputs=inputs, extra=extra)
        self._log_run_start(chain_run)

    def log_chain_run_end(self, outputs: Dict[str, Any], error=None) -> None:
        """Log the end of a chain run."""

        if not self._stack:
            raise LoggerException("No ChainRun found to be logged")

        chain_run = self._stack.pop()
        if not isinstance(chain_run, ChainRun):
            self._session.rollback()
            raise LoggerException("ChainRun end can only be logged after a ChainRun start")
        chain_run.outputs = outputs
        chain_run.error = error
        chain_run.end_time = datetime.datetime.utcnow()
        chain_run.execution_order = self._execution_order
        self._end_log_run()

    def log_tool_run_start(self, serialized: Dict[str, Any], action: str, inputs: Dict[str, Any], **extra: str) -> None:
        """Log the start of a tool run."""

        tool_run = ToolRun(serialized=serialized, action=action, inputs=inputs, extra=extra)
        self._log_run_start(tool_run)

    def log_tool_run_end(self, outputs: Dict[str, Any], error=None) -> None:
        """Log the end of a tool run."""

        if not self._stack:
            raise LoggerException("No ToolRun found to be logged")

        tool_run = self._stack.pop()
        if not isinstance(tool_run, ToolRun):
            self._session.rollback()
            raise LoggerException("ToolRun end can only be logged after a ToolRun start")
        tool_run.outputs = outputs
        tool_run.error = error
        tool_run.end_time = datetime.datetime.utcnow()
        tool_run.execution_order = self._execution_order
        self._end_log_run()

    def get_llm_runs(self) -> List[base.LLMRun]:
        """Return all the LLM runs."""

        llm_runs = self._session.scalars(select(LLMRun)).all()
        return [_convert_llm_run(llm_run) for llm_run in llm_runs]

    # TODO: specify nesting, utilize joined loads
    def get_chain_runs(self) -> List[base.ChainRun]:
        """Return all the chain runs."""

        chain_runs = self._session.scalars(select(ChainRun)).all()
        return [_deep_convert_run(chain_run) for chain_run in chain_runs]

    # TODO: specify nesting, utilize joined loads
    def get_tool_runs(self) -> List[base.ToolRun]:
        """Return all the tool runs."""

        tool_runs = self._session.scalars(select(ToolRun)).all()
        return [_deep_convert_run(tool_run) for tool_run in tool_runs]

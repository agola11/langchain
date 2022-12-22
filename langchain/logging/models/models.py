from sqlalchemy import create_engine
from sqlalchemy.exc import DatabaseError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
import os
from typing import Any
import datetime
from typing import Any, Dict, List, Tuple, Type, Union

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Table,
    and_,
    create_engine,
    select,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Session, declarative_base, declarative_mixin, relationship

from langchain.logging import base
from langchain.logging.base import BaseLogger


if not os.environ.get("DATABASE_URL"):
    engine = create_engine("sqlite://")
else:
    engine = create_engine(os.environ.get("DATABASE_URL"))

# Use thread-local session
session = scoped_session(sessionmaker(autocommit=False, bind=engine))


def _flush():
    """Flush the session."""

    try:
        session.flush()
    except DatabaseError:
        session.rollback()
        raise


class RunBase:
    """Base class for all runs."""

    def save(self) -> "RunBase":
        """Save the run."""

        session.add(self)
        _flush()
        return self

    def update(self, **kwargs: Any) -> "RunBase":
        """Update the run."""

        for attr, value in kwargs.items():
            setattr(self, attr, value)
        return self.save()

    def delete(self):
        """Delete the run."""

        session.delete(self)
        _flush()


RunModel = declarative_base(cls=RunBase)
RunModel.query = session.query_property()


@declarative_mixin
class RunMixin:
    """Mixin for all runs."""

    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, default=datetime.datetime.utcnow)
    extra = Column(JSON, default=None)
    error = Column(JSON, default=None)
    execution_order = Column(Integer, default=1)
    serialized = Column(JSON)


class LLMRun(RunModel, RunMixin):
    """LLM run model."""

    __tablename__ = "llm_runs"

    id = Column(Integer, primary_key=True)
    prompts = Column(JSON)
    response = Column(JSON)

    parent_chain_run_id = Column(Integer, ForeignKey("chain_runs.id"))
    parent_chain_run = relationship("ChainRun", back_populates="child_llm_runs")

    parent_tool_run_id = Column(Integer, ForeignKey("tool_runs.id"))
    parent_tool_run = relationship("ToolRun", back_populates="child_llm_runs")

    def __repr__(self):
        return f"<LLMRun(serialized={self.serialized}, execution_order={self.execution_order}, id={self.id}, prompts={self.prompts}, response={self.response})>"


ct_association_table = Table(
    "chain_runs_tool_runs",
    RunModel.metadata,
    Column("parent_chain_run_id", ForeignKey("chain_runs.id")),
    Column("child_tool_run_id", ForeignKey("tool_runs.id")),
)

tc_association_table = Table(
    "tool_runs_chain_runs",
    RunModel.metadata,
    Column("parent_tool_run_id", ForeignKey("tool_runs.id")),
    Column("child_chain_run_id", ForeignKey("chain_runs.id")),
)


class ChainRun(RunModel, RunMixin):
    """Chain run model."""

    __tablename__ = "chain_runs"

    id = Column(Integer, primary_key=True)
    inputs = Column(JSON)
    outputs = Column(JSON)

    child_llm_runs = relationship("LLMRun", back_populates="parent_chain_run")

    parent_chain_run_id = Column(Integer, ForeignKey("chain_runs.id"))
    parent_chain_run = relationship(
        "ChainRun", remote_side=[id], backref="child_chain_runs"
    )

    child_tool_runs = relationship(
        "ToolRun", secondary=ct_association_table, back_populates="parent_chain_run"
    )
    parent_tool_run = relationship(
        "ToolRun", secondary=tc_association_table, back_populates="child_chain_runs"
    )

    def __repr__(self):
        return f"<ChainRun(serialized={self.serialized}, execution_order={self.execution_order}, id={self.id}, inputs={self.inputs}, outputs={self.outputs})>"


class ToolRun(RunModel, RunMixin):
    """Tool run model."""

    __tablename__ = "tool_runs"

    id = Column(Integer, primary_key=True)
    action = Column(String)
    inputs = Column(JSON)
    outputs = Column(JSON)

    child_llm_runs = relationship("LLMRun", back_populates="parent_tool_run")

    parent_tool_run_id = Column(Integer, ForeignKey("tool_runs.id"))
    parent_tool_run = relationship(
        "ToolRun", remote_side=[id], backref="child_tool_runs"
    )

    child_chain_runs = relationship(
        "ChainRun", secondary=tc_association_table, back_populates="parent_tool_run"
    )
    parent_chain_run = relationship(
        "ChainRun", secondary=ct_association_table, back_populates="child_tool_runs"
    )

    def __repr__(self):
        return f"<ToolRun(serialized={self.serialized}, execution_order={self.execution_order}, inputs={self.inputs}, outputs={self.outputs})>"


RunModel.metadata.create_all(engine)


def _print_run(run: Union[LLMRun, ChainRun, ToolRun], tabs: str) -> None:
    """Pretty print a run."""

    print(f"{tabs}Run: {run}")
    if isinstance(run, ChainRun) or isinstance(run, ToolRun):
        all_runs = run.child_chain_runs + run.child_llm_runs + run.child_tool_runs
        all_runs.sort(key=lambda x: x.execution_order)
        for child_run in all_runs:
            _print_run(child_run, tabs + "\t")


def main() -> None:
    """Run example database operations."""

    zero_shot_run = ChainRun(
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
                        execution_order=1,
                    )
                ],
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
                ],
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
                ],
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
                ],
            ),
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
                                        execution_order=9,
                                    )
                                ],
                            )
                        ],
                    )
                ],
            ),
        ],
    )
    session.add(zero_shot_run)
    session.commit()

    from sqlalchemy.orm import joinedload

    stmt = (
        select(ChainRun)
        .where(ChainRun.id == 1)
        .options(
            joinedload(ChainRun.child_llm_runs),
            joinedload(ChainRun.child_chain_runs).joinedload(ChainRun.child_llm_runs),
            joinedload(ChainRun.child_tool_runs),
        )
    )
    zeroshot_chain = session.scalars(stmt).unique().one()
    _print_run(zeroshot_chain, "")


if __name__ == "__main__":
    main()

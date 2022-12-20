from sqlalchemy import Column, ForeignKey, Integer, Table, DateTime, String, Boolean
from sqlalchemy.orm import declarative_base, relationship
from typing import Union

from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_mixin
import datetime

Base = declarative_base()


# Assumptions:
# 1. A chain run can have multiple LLM runs, chain runs, and tool runs
# 2. A tool run can have multiple LLM runs, chain runs, and tool runs
# 3. An LLM cannot have any child runs

@declarative_mixin
class RunMixin:
    run_date = Column(DateTime, default=datetime.datetime.utcnow)
    success = Column(Boolean, default=True)
    extra = Column(JSON, default={})
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
        return f"<LLMRun(id={self.id}, prompts={self.prompts}, response={self.response}, execution_order={self.execution_order})>"


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
        return f"<ChainRun(id={self.id}, inputs={self.inputs}, outputs={self.outputs}, execution_order={self.execution_order})>"


class ToolRun(Base, RunMixin):
    __tablename__ = "tool_runs"

    id = Column(Integer, primary_key=True)
    tool_action = Column(String)
    inputs = Column(JSON)
    outputs = Column(JSON)

    child_llm_runs = relationship("LLMRun", back_populates="tool_run")

    parent_tool_run_id = Column(Integer, ForeignKey("tool_runs.id"))
    parent_tool_run = relationship("ToolRun", remote_side=[id], backref="child_tool_runs")

    child_chain_runs = relationship("ChainRun", secondary=tc_association_table, back_populates="parent_tool_run")
    parent_chain_run = relationship("ChainRun", secondary=ct_association_table, back_populates="child_tool_runs")

    def __repr__(self):
        return f"<ToolRun(id={self.id}, tool_action={self.tool_action}, inputs={self.inputs}, outputs={self.outputs}, execution_order={self.execution_order})>"


def main() -> None:
    from sqlalchemy import create_engine
    # engine = create_engine("sqlite:///log.db", echo=True, future=True) # Persistent
    engine = create_engine("sqlite://", echo=False, future=True)  # In-Memory
    Base.metadata.create_all(engine)

    from sqlalchemy.orm import Session
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
                    tool_action="SEARCH",
                    inputs={"text": "Search input1"},
                    outputs={"response": "Search output1"},
                    serialized={"name": "SerpAPIWrapper"},
                    execution_order=3,
                ),
                ToolRun(
                    tool_action="SEARCH",
                    inputs={"text": "Search input2"},
                    outputs={"response": "Search output2"},
                    serialized={"name": "SerpAPIWrapper"},
                    execution_order=6,
                ),
                ToolRun(
                    tool_action="CALCULATOR",
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
    from sqlalchemy import select

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


if __name__ == "__main__":
    main()



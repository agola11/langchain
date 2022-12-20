from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
# Load the tool configs that are needed.
from langchain import LLMMathChain, SerpAPIWrapper
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import declarative_mixin
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from langchain.logging.sqlite import ChainRun, LLMRun, ToolRun, print_run, SqliteLogger
from sqlalchemy.orm import joinedload
from sqlalchemy import select


def main():
    llm = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    agent.run("Who won the US Open men's tennis final in 2019? What is his age raised to the second power??")
    # agent.run("Who won the US Open men's tennis final in 2019? What is the next prime number after his age?")
    # agent.run("Who won the US Open men's tennis final in 2022? What is the next prime number after his age?")

    # TODO: need to create log retrieval functions to do this so we don't need to grab the session directly
    session = SqliteLogger()._session
    stmt = select(ChainRun).where(ChainRun.id == 1).options(joinedload(ChainRun.child_llm_runs),
                                                            joinedload(ChainRun.child_chain_runs).joinedload(
                                                                ChainRun.child_llm_runs),
                                                            joinedload(ChainRun.child_tool_runs))
    zeroshot_chain = session.scalars(stmt).unique().one()
    print_run(zeroshot_chain, "")

    agent.run("Who won the US Open men's tennis final in 2022? What is his age raised to the third power??")
    stmt = select(ChainRun).where(ChainRun.id == 8).options(joinedload(ChainRun.child_llm_runs),
                                                            joinedload(ChainRun.child_chain_runs).joinedload(
                                                                ChainRun.child_llm_runs),
                                                            joinedload(ChainRun.child_tool_runs))
    zeroshot_chain = session.scalars(stmt).unique().one()
    print_run(zeroshot_chain, "")


if __name__ == "__main__":
    main()

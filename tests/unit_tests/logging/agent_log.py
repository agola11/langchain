from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.logging.sqlite import print_base_run
from langchain.logging import get_logger


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
    agent.run("Who won the US Open men's tennis final in 2022? What is his age raised to the third power??")

    chain_runs = get_logger().get_chain_runs()
    print_base_run(chain_runs[0], "")
    print("___________________________")
    print_base_run(chain_runs[1], "")
    print("___________________________")
    print_base_run(chain_runs[7], "")


if __name__ == "__main__":
    main()

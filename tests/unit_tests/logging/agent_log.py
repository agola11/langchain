from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import Tool, initialize_agent
from langchain.llms import OpenAI
from langchain.logging import get_logger


def main():
    llm = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="useful for when you need to answer questions about current events",
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math",
        ),
    ]

    agent = initialize_agent(
        tools, llm, agent="zero-shot-react-description", verbose=False
    )
    # agent.run(
    #     "Who won the US Open men's tennis final in 2019? What is his age raised to the second power??"
    # )
    # agent.run("Who won the US Open men's tennis final in 2019? What is the next prime number after his age?")
    # agent.run("Who won the US Open men's tennis final in 2022? What is the next prime number after his age?")
    # agent.run(
    #     "Who won the US Open men's tennis final in 2022? What is his age raised to the third power??"
    # )

    import threading

    inputs = [
        "Who won the US Open men's tennis final in 2019? What is his age raised to the second power??",
        "Who won the US Open men's tennis final in 2022? What is his age raised to the third power??",
        "Who won the US Open men's tennis final in 2022? What is his age raised to the second power??",
        "Who won the US Open men's tennis final in 2022? What is his age raised to the fourth power??"
    ]

    threads = list()
    for index in range(5):
        print(f"Main    : create and start thread {index}")
        x = threading.Thread(target=agent.run, args=(inputs[0],))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        print(f"Main    : before joining thread {index}.")
        thread.join()
        print(f"Main    : thread {index} done")

    chain_runs = get_logger().get_chain_runs(top_level_only=True)
    all_chain_runs = get_logger().get_chain_runs()
    print(chain_runs[0].to_json(indent=2))
    print("___________________________")
    print(chain_runs[1].to_json(indent=2))
    print("___________________________")

    print(f"Got {len(chain_runs)} top level chain runs")
    print(f"Got {len(all_chain_runs)} inclusive chain runs")
    #
    # print(get_logger().get_chain_run(chain_runs[0].id).to_json(indent=2))


if __name__ == "__main__":
    main()

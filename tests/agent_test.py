from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import os

PROMPT = """
You are GPT-3, and you can't do math.

You can do basic math, and your memorization abilities are impressive, but you can't do any complex calculations that a human could not do in their head. You also have an annoying tendency to just make up highly specific, but wrong, answers.

So we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math problem, just use this format and we’ll take care of the rest:

Question: ${Question with hard calculation.}
```python
${Code that prints what you need to know}
```
```output
${Output of your code}
```
Answer: ${Answer}

Otherwise, use this simpler format:

Question: ${Question without hard calculation}
Answer: ${Answer}

Begin.

Question: What is 37593 * 67?

```python
print(37593 * 67)
```
```output
2518731
```
Answer: 2518731

Question: 19
"""

# Load the tool configs that are needed.
from langchain import LLMMathChain, SerpAPIWrapper

def main():
    llm = OpenAI(temperature=0)
    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool(
            name = "Search",
            func=search.run,
            description="useful for when you need to answer questions about current events"
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about math"
        )
    ]

    # print(llm.generate([PROMPT], stop=["```output"]))

    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=False)
    agent.run("Who won the US Open men's tennis final in 2019? What is his age raised to the second power??")
    #agent.run("Who won the US Open men's tennis final in 2019? What is the next prime number after his age?")
    #agent.run("Who won the US Open men's tennis final in 2022? What is the next prime number after his age?")



if __name__ == "__main__":
    main()

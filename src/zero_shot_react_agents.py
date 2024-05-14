import os

import openai
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.agents.load_tools import load_tools
from langchain_openai.llms.base import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Set your API Key from OpenAI
openai_api_key = "<OPENAI_API_TOKEN>"

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore

# Define the tools
tools = load_tools(["llm-math"], llm=llm)

# Define the agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Run the agent
agent.run("What is 10 multiplied by 50?")

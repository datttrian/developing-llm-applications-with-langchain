import os

import openai
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain_core.tools import Tool, tool
from langchain_openai import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Define the calculate_ltv tool function
@tool
def calculate_ltv(company_name: str) -> str:
    """Generate the LTV for a company."""
    avg_churn = 0.25
    avg_revenue = 1000
    historical_LTV = avg_revenue / avg_churn

    report = f"LTV Report for {company_name}\n"
    report += f"Avg. churn: ${avg_churn}\n"
    report += f"Avg. revenue: ${avg_revenue}\n"
    report += f"historical_LTV: ${historical_LTV}\n"
    return report


# Define the tools list
tools = [
    Tool(
        name="LTVReport",
        func=calculate_ltv,
        description="Use this for calculating historical LTV.",
    )
]

# Initialize the appropriate agent type
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("Run a financial report that calculates historical LTV for Hooli")

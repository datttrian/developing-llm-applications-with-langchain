import os

import openai
from dotenv import load_dotenv
from langchain_community.tools.convert_to_openai import format_tool_to_openai_function
from langchain_core.tools import tool
from pydantic.v1.main import BaseModel, Field

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Create an LTVDescription class to manually add a function description
class LTVDescription(BaseModel):
    query: str = Field(description="Calculate an extremely simple historical LTV")


# Format the calculate_ltv tool function so it can be used by OpenAI models
@tool(args_schema=LTVDescription)  # type: ignore
def calculate_ltv(company_name: str) -> str:  # pylint: disable=function-redefined
    """Generate the LTV for a company to pontificate with."""
    avg_churn = 0.25
    avg_revenue = 1000
    historical_LTV = avg_revenue / avg_churn

    report = f"Pontification Report for {company_name}\n"
    report += f"Avg. churn: ${avg_churn}\n"
    report += f"Avg. revenue: ${avg_revenue}\n"
    report += f"historical_LTV: ${historical_LTV}\n"
    return report


print(format_tool_to_openai_function(calculate_ltv))

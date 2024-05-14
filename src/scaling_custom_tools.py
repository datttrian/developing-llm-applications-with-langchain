import os

import openai
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


def calculate_wellness_score(
    sleep_hours, exercise_minutes, healthy_meals, stress_level
):
    """Calculate a Wellness Score based on sleep, exercise, nutrition, and stress management."""
    max_score_per_category = 25

    sleep_score = min(sleep_hours / 8 * max_score_per_category, max_score_per_category)
    exercise_score = min(
        exercise_minutes / 30 * max_score_per_category, max_score_per_category
    )
    nutrition_score = min(
        healthy_meals / 3 * max_score_per_category, max_score_per_category
    )
    stress_score = max_score_per_category - min(
        stress_level / 10 * max_score_per_category, max_score_per_category
    )

    total_score = sleep_score + exercise_score + nutrition_score + stress_score
    return total_score


# Create a structured tool from calculate_wellness_score()
tools = [StructuredTool.from_function(calculate_wellness_score)]  # type: ignore

# Initialize the appropriate agent type and tool set
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

wellness_tool = tools[0]
result = wellness_tool.func(
    sleep_hours=8, exercise_minutes=14, healthy_meals=10, stress_level=20  # type: ignore
)
print(result)

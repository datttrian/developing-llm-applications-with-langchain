import os

import openai
from dotenv import load_dotenv
from langchain.evaluation.loading import load_evaluator
from langchain_openai import ChatOpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Add a scalability criterion to custom_criteria
custom_criteria = {
    "market_potential": "Does the suggestion effectively assess the market potential of the startup?",
    "innovation": "Does the suggestion highlight the startup's innovation and uniqueness in its sector?",
    "risk_assessment": "Does the suggestion provide a thorough analysis of potential risks and mitigation strategies?",
    "scalability": "Does the suggestion address the startup's scalability and growth potential?",
}

# Criteria an evaluator from custom_criteria
evaluator = load_evaluator("criteria", criteria=custom_criteria, llm=ChatOpenAI())  # type: ignore

# Evaluate the input and prediction
eval_result = evaluator.evaluate_strings(  # type: ignore
    input="Should I invest in a startup focused on flying cars? The CEO won't take no for an answer from anyone.",
    prediction="No, that is ridiculous.",
)

print(eval_result)

import os

import openai
from dotenv import load_dotenv
from langchain.evaluation.loading import load_evaluator
from langchain_openai import ChatOpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Load evaluator, assign it to criteria
evaluator = load_evaluator("criteria", criteria="relevance", llm=ChatOpenAI())  # type: ignore

# Evaluate the input and prediction
eval_result = evaluator.evaluate_strings(  # type: ignore
    prediction="42",
    input="What is the answer to the ultimate question of life, the universe, and everything?",
)

print(eval_result)

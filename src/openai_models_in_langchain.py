import os

import openai
from dotenv import load_dotenv
from langchain_openai import OpenAI


# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define the LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")  # type: ignore

# Predict the words following the text in question
question = "Whatever you do, take care of your shoes"
output = llm.invoke(question)

print(output)

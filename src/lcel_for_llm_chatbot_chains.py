import os

import openai
from dotenv import load_dotenv
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template(
    "You are a skilled poet. Write a haiku about the following topic: {topic}"
)

# Define the chain using LCEL
chain = prompt | model

# Invoke the chain with any topic
print(chain.invoke({"topic": "Large Language Models"}))

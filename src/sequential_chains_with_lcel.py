import os

import openai
from dotenv import load_dotenv
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

coding_prompt = PromptTemplate.from_template(
    """Write Python code to loop through the following list, printing each element: {list}"""
)
validate_prompt = PromptTemplate.from_template(
    """Consider the following Python code: {answer} If it doesn't use a list comprehension, update it to use one. If it does use a list comprehension, return the original code without explanation:"""
)

llm = ChatOpenAI()

# Create the sequential chain
chain = (
    {"answer": coding_prompt | llm | StrOutputParser()}
    | validate_prompt
    | llm
    | StrOutputParser()  # type: ignore
)

# Invoke the chain with the user's question
chain.invoke({"list": "[3, 1, 4, 1]"})

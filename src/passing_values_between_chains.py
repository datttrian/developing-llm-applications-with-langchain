import os

import openai
from dotenv import load_dotenv
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Make ceo_response available for other chains
ceo_response = (
    ChatPromptTemplate.from_template(
        "You are a CEO. Describe the most lucrative consumer product addressing the following consumer need in one sentence: {input}."  # type: ignore
    )
    | ChatOpenAI()
    | {"ceo_response": RunnablePassthrough() | StrOutputParser()}
)

advisor_response = (
    ChatPromptTemplate.from_template(
        "You are a strategic adviser. Briefly map the outline and business plan for {ceo_response} in 3 key steps."
    )
    | ChatOpenAI()
    | StrOutputParser()
)

overall_response = (
    ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "CEO response:\n{ceo_response}\n\nAdvisor response:\n{advisor_response}",
            ),
            (
                "system",
                "Generate a final response including the CEO's response, the advisor response, and a summary of the business plan in one sentence.",
            ),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

# Create a chain to insert the outputs from the other chains into overall_response
business_idea_chain = (
    {"ceo_response": ceo_response, "advisor_response": advisor_response}  # type: ignore
    | overall_response
    | ChatOpenAI()
    | StrOutputParser()
)

print(
    business_idea_chain.invoke(
        {
            "input": "Typing on mobile touchscreens is slow.",
            "ceo_response": "",
            "advisor_response": "",
        }
    )
)

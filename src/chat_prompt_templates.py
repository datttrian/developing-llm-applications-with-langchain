import os

import openai
from dotenv import load_dotenv
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_openai.chat_models.base import ChatOpenAI


# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Define an OpenAI chat model
llm = ChatOpenAI(temperature=0)

# Create a chat prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Respond to question: {question}"),
    ]
)

# Insert a question into the template and call the model
full_prompt = prompt_template.format_messages(question="How can I retain learning?")
print(llm(full_prompt))

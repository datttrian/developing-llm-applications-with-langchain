import os

import openai
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory.summary import ConversationSummaryMemory
from langchain_openai import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
chat = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore

# Define a summary memory that uses an OpenAI model
memory = ConversationSummaryMemory(llm=OpenAI(model_name="gpt-3.5-turbo-instruct"))  # type: ignore

# Define the chain for integrating the memory with the model
summary_chain = ConversationChain(llm=chat, memory=memory, verbose=True)

# Invoke the chain with the inputs provided
summary_chain.predict(
    input="Describe the relationship of the human mind with the keyboard when taking a great online class."
)
summary_chain.predict(input="Use an analogy to describe it.")

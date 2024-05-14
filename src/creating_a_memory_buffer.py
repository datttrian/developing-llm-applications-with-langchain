import os

import openai
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_openai import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
chat = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore

# Define a buffer memory
memory = ConversationBufferMemory(size=4)  # type: ignore

# Define the chain for integrating the memory with the model
buffer_chain = ConversationChain(llm=chat, memory=memory, verbose=True)

# Invoke the chain with the inputs provided
buffer_chain.predict(input="Write Python code to draw a scatter plot.")
buffer_chain.predict(input="Use the Seaborn library.")

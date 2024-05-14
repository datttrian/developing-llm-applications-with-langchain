import os

import openai
from dotenv import load_dotenv
from langchain_community.chat_message_histories.in_memory import ChatMessageHistory
from langchain_openai.chat_models.base import ChatOpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]
chat = ChatOpenAI(temperature=0)

# Create the conversation history and add the first AI message
history = ChatMessageHistory()
history.add_ai_message("Hello! Ask me anything about Python programming!")

# Add the user message to the history and call the model
history.add_user_message("What is a list comprehension?")
ai_response = chat(history.messages)
print(ai_response)

# Add another user message and call the model
history.add_user_message("Describe the same in fewer words")
ai_response = chat(history.messages)
print(ai_response)

import os

import openai
from dotenv import load_dotenv
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain.memory.summary import ConversationSummaryMemory
from langchain_community.chat_message_histories.in_memory import \
    ChatMessageHistory
from langchain_community.llms import \
    HuggingFaceHub  # pylint: disable=no-name-in-module.
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI
from langchain_openai.chat_models.base import ChatOpenAI

load_dotenv()


# Set your Hugging Face API token
huggingfacehub_api_token = os.environ["HUGGINGFACE_API_KEY"]

# Define the LLM
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=huggingfacehub_api_token,
)

# Predict the words following the text in question
question = "Whatever you do, take care of your shoes"
output = llm.invoke(question)

print(output)


# Set your API Key from OpenAI
openai.api_key = os.environ["OPENAI_API_KEY"]
# Define the LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")  # type: ignore

# Predict the words following the text in question
question = "Whatever you do, take care of your shoes"
output = llm.invoke(question)

print(output)


# Set your Hugging Face API token
huggingfacehub_api_token = os.environ["HUGGINGFACE_API_KEY"]

# Create a prompt template from the template string
template = (
    "You are an artificial intelligence assistant, answer the question. {question}"
)
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=huggingfacehub_api_token,
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How does LangChain make LLM application development easier?"
print(llm_chain.run(question))


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
llm(full_prompt)


# Set your API Key from OpenAI
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


# Set your API Key from OpenAI
chat = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore

# Define a buffer memory
memory = ConversationBufferMemory(size=4)  # type: ignore

# Define the chain for integrating the memory with the model
buffer_chain = ConversationChain(llm=chat, memory=memory, verbose=True)

# Invoke the chain with the inputs provided
buffer_chain.predict(input="Write Python code to draw a scatter plot.")
buffer_chain.predict(input="Use the Seaborn library.")


# Set your API Key from OpenAI
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

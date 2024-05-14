import os

import openai
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

# Create the retriever and model
vectorstore = Chroma.from_texts(
    ["LangChain v0.1.0 was released on January 8, 2024."], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
model = ChatOpenAI(temperature=0)

template = (
    """Answer the question based on the context:{context}. Question: {question}"""
)
prompt = ChatPromptTemplate.from_template(template)

# Create the chain and run it
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model  # type: ignore

chain.invoke("When was LangChain v0.1.0 released?")  # type: ignore

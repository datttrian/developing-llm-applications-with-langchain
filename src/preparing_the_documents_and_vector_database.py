import os

import openai
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings.base import OpenAIEmbeddings

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFLoader("attention_is_all_you_need.pdf")
data = loader.load()
chunk_size = 200
chunk_overlap = 50

# Split the quote using RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap  # type: ignore
)
docs = splitter.split_documents(data)  # type: ignore

# Define an OpenAI embeddings model
embedding_model = OpenAIEmbeddings()

# Create the Chroma vector DB using the OpenAI embedding function; persist the database
vectordb = Chroma(
    persist_directory="embedding/chroma/", embedding_function=embedding_model
)
vectordb.persist()

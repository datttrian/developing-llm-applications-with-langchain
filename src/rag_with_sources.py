import os

import openai
from dotenv import load_dotenv
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai.llms.base import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

loader = PyPDFLoader("attention_is_all_you_need.pdf")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50, separators=["."]  # type: ignore
)
docs = splitter.split_documents(data)  # type: ignore

embedding_model = OpenAIEmbeddings()
docstorage = Chroma.from_documents(docs, embedding_model)  # type: ignore

# Define the function for the question to be answered with
qa = RetrievalQAWithSourcesChain.from_chain_type(
    OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0),  # type: ignore
    chain_type="stuff",
    retriever=docstorage.as_retriever(),
)

# Run the query on the documents
results = qa(
    {"question": "What is the primary architecture presented in the document?"},
    return_only_outputs=True,
)
print(results)

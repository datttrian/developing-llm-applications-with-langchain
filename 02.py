# Import library
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import HNLoader, PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai.embeddings.base import OpenAIEmbeddings
from langchain_openai.llms.base import OpenAI


# Create a document loader for attention_is_all_you_need.pdf
loader = PyPDFLoader("attention_is_all_you_need.pdf")

# Load the document
data = loader.load()
print(data[0])


# Import library


# Create a document loader for fifa_countries_audience.csv
loader = CSVLoader(file_path="fifa_countries_audience.csv")  # type: ignore

# Load the document
data = loader.load()
print(data[0])


# Create a document loader for the top Hacker News stories
loader = HNLoader("https://news.ycombinator.com")  # type: ignore

# Load the document
data = loader.load()

# Print the first document
print(data[0])

# Print the first document's metadata
print(data[0].metadata)


# Import libary


quote = "One machine can do the work of fifty ordinary humans. No machine can do the work of one extraordinary human."
chunk_size = 24
chunk_overlap = 3

# Create an instance of the splitter class
splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Split the document and print the chunks
docs = splitter.split_text(quote)
print(docs)


# Import libary


quote = "Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe."
chunk_size = 24
chunk_overlap = 10

# Create an instance of the splitter class
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap  # type: ignore
)

# Split the document and print the chunks
docs = splitter.split_text(quote)
print(docs)


# Load the HTML document into memory
loader = UnstructuredHTMLLoader("white_house_executive_order_nov_2023.html")  # type: ignore
data = loader.load()

# Define variables
chunk_size = 300
chunk_overlap = 100

# Split the HTML
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["."]  # type: ignore
)

docs = splitter.split_documents(data)  # type: ignore
print(docs)


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


loader = PyPDFLoader("attention_is_all_you_need.pdf")
data = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50, separators=["."]  # type: ignore
)
docs = splitter.split_documents(data)  # type: ignore

# Embed the documents and store them in a Chroma DB
embedding_model = OpenAIEmbeddings()
docstorage = Chroma.from_documents(docs, embedding_model)  # type: ignore

# Define the Retrieval QA Chain to integrate the database and LLM
qa = RetrievalQA.from_chain_type(
    OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0),  # type: ignore
    chain_type="stuff",
    retriever=docstorage.as_retriever(),
)

# Run the chain on the query provided
query = "What is the primary architecture presented in the document?"
qa.run(query)


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

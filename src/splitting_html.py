from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.html import UnstructuredHTMLLoader

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

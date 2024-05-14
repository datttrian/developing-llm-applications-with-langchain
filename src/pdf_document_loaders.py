from langchain_community.document_loaders import PyPDFLoader

# Create a document loader for attention_is_all_you_need.pdf
loader = PyPDFLoader("attention_is_all_you_need.pdf")

# Load the document
data = loader.load()
print(data[0])

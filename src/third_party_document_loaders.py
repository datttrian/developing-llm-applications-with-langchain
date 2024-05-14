from langchain_community.document_loaders import HNLoader

# Create a document loader for the top Hacker News stories
loader = HNLoader("https://news.ycombinator.com")  # type: ignore

# Load the document
data = loader.load()

# Print the first document
print(data[0])

# Print the first document's metadata
print(data[0].metadata)

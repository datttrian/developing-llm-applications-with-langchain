from langchain.text_splitter import RecursiveCharacterTextSplitter

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

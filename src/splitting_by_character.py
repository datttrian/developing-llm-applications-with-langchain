from langchain.text_splitter import CharacterTextSplitter

quote = "One machine can do the work of fifty ordinary humans. No machine can do the work of one extraordinary human."
chunk_size = 24
chunk_overlap = 3

# Create an instance of the splitter class
splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Split the document and print the chunks
docs = splitter.split_text(quote)
print(docs)

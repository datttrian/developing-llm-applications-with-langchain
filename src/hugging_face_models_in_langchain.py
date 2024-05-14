import os

from dotenv import load_dotenv
from langchain_community.llms import HuggingFaceHub  # pylint: disable=no-name-in-module

# Set your Hugging Face API token
load_dotenv()
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

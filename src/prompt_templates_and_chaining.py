import os

from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_community.llms import HuggingFaceHub  # pylint: disable=no-name-in-module
from langchain_core.prompts.prompt import PromptTemplate


# Set your Hugging Face API token
load_dotenv()
huggingfacehub_api_token = os.environ["HUGGINGFACE_API_KEY"]

# Create a prompt template from the template string
template = (
    "You are an artificial intelligence assistant, answer the question. {question}"
)
prompt = PromptTemplate(template=template, input_variables=["question"])

# Create a chain to integrate the prompt template and LLM
llm = HuggingFaceHub(
    repo_id="tiiuae/falcon-7b-instruct",
    huggingfacehub_api_token=huggingfacehub_api_token,
)
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How does LangChain make LLM application development easier?"
print(llm_chain.run(question))

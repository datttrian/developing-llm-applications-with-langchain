import os

import openai
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Complete the CallingItIn class to return the prompt, model_name, and temperature
class CallingItIn(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, invocation_params, **kwargs):
        print(prompts)
        print(invocation_params["model_name"])
        print(invocation_params["temperature"])


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", streaming=True)  # type: ignore
prompt_template = "What do {animal} like to eat?"
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

# Call the model with the parameters needed by the prompt
output = chain.run({"animal": "wombats"}, callbacks=[CallingItIn()])
print(output)

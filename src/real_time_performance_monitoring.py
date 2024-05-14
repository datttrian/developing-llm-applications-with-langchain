import os
import time

import openai
from dotenv import load_dotenv
from langchain.chains.llm import LLMChain
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import OpenAI

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


# Complete the PerformanceMonitoringCallback class to return the token and time
class PerformanceMonitoringCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(f"Token: {repr(token)} generated at time: {time.time()}")


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0, streaming=True)  # type: ignore
prompt_template = "Describe the process of photosynthesis."
chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

# Call the chain with the callback
output = chain.run({}, callbacks=[PerformanceMonitoringCallback()])
print("Final Output:", output)

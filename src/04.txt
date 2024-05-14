import time

from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.chains.llm import LLMChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.evaluation.loading import load_evaluator
from langchain.evaluation.qa.eval_chain import QAEvalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools.convert_to_openai import \
    format_tool_to_openai_function
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.tools import StructuredTool, Tool, tool
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic.v1.main import BaseModel, Field


# Define the calculate_ltv tool function
@tool
def calculate_ltv(company_name: str) -> str:
    """Generate the LTV for a company."""
    avg_churn = 0.25
    avg_revenue = 1000
    historical_LTV = avg_revenue / avg_churn

    report = f"LTV Report for {company_name}\n"
    report += f"Avg. churn: ${avg_churn}\n"
    report += f"Avg. revenue: ${avg_revenue}\n"
    report += f"historical_LTV: ${historical_LTV}\n"
    return report


# Define the tools list
tools = [
    Tool(
        name="LTVReport",
        func=calculate_ltv,
        description="Use this for calculating historical LTV.",
    )
]

# Initialize the appropriate agent type
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
agent.run("Run a financial report that calculates historical LTV for Hooli")


def calculate_wellness_score(
    sleep_hours, exercise_minutes, healthy_meals, stress_level
):
    """Calculate a Wellness Score based on sleep, exercise, nutrition, and stress management."""
    max_score_per_category = 25

    sleep_score = min(sleep_hours / 8 * max_score_per_category, max_score_per_category)
    exercise_score = min(
        exercise_minutes / 30 * max_score_per_category, max_score_per_category
    )
    nutrition_score = min(
        healthy_meals / 3 * max_score_per_category, max_score_per_category
    )
    stress_score = max_score_per_category - min(
        stress_level / 10 * max_score_per_category, max_score_per_category
    )

    total_score = sleep_score + exercise_score + nutrition_score + stress_score
    return total_score


# Create a structured tool from calculate_wellness_score()
tools = [StructuredTool.from_function(calculate_wellness_score)]  # type: ignore

# Initialize the appropriate agent type and tool set
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

wellness_tool = tools[0]
result = wellness_tool.func(
    sleep_hours=8, exercise_minutes=14, healthy_meals=10, stress_level=20  # type: ignore
)
print(result)


# Create an LTVDescription class to manually add a function description
class LTVDescription(BaseModel):
    query: str = Field(description="Calculate an extremely simple historical LTV")


# Format the calculate_ltv tool function so it can be used by OpenAI models
@tool(args_schema=LTVDescription)  # type: ignore
def calculate_ltv(company_name: str) -> str:  # pylint: disable=function-redefined
    """Generate the LTV for a company to pontificate with."""
    avg_churn = 0.25
    avg_revenue = 1000
    historical_LTV = avg_revenue / avg_churn

    report = f"Pontification Report for {company_name}\n"
    report += f"Avg. churn: ${avg_churn}\n"
    report += f"Avg. revenue: ${avg_revenue}\n"
    report += f"historical_LTV: ${historical_LTV}\n"
    return report


print(format_tool_to_openai_function(calculate_ltv))  # type: ignore


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


# Load evaluator, assign it to criteria
evaluator = load_evaluator("criteria", criteria="relevance", llm=ChatOpenAI())  # type: ignore

# Evaluate the input and prediction
eval_result = evaluator.evaluate_strings(  # type: ignore
    prediction="42",
    input="What is the answer to the ultimate question of life, the universe, and everything?",
)

print(eval_result)


# Add a scalability criterion to custom_criteria
custom_criteria = {
    "market_potential": "Does the suggestion effectively assess the market potential of the startup?",
    "innovation": "Does the suggestion highlight the startup's innovation and uniqueness in its sector?",
    "risk_assessment": "Does the suggestion provide a thorough analysis of potential risks and mitigation strategies?",
    "scalability": "Does the suggestion address the startup's scalability and growth potential?",
}

# Criteria an evaluator from custom_criteria
evaluator = load_evaluator("criteria", criteria=custom_criteria, llm=ChatOpenAI())  # type: ignore

# Evaluate the input and prediction
eval_result = evaluator.evaluate_strings(  # type: ignore
    input="Should I invest in a startup focused on flying cars? The CEO won't take no for an answer from anyone.",
    prediction="No, that is ridiculous.",
)

print(eval_result)


question_set = [
    {
        "question": "What is the primary architecture presented in the document?",
        "answer": "The Transformer.",
    },
    {
        "question": "According to the document, is the Transformer faster or slower than architectures based on recurrent or convolutional layers?",
        "answer": "The Transformer is faster.",
    },
    {
        "question": "Who is the primary author of the document?",
        "answer": "Ashish Vaswani.",
    },
]


embedding = OpenAIEmbeddings()
loader = PyPDFLoader("attention_is_all_you_need.pdf")
data = loader.load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200, chunk_overlap=50, separators=["."]  # type: ignore
)
docs = splitter.split_documents(data)  # type: ignore
docstorage = Chroma.from_documents(docs, embedding)
llm = OpenAI(model_name="gpt-3.5-turbo-instruct")  # type: ignore

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docstorage.as_retriever(),
    input_key="question",
)

# Generate the model responses using the RetrievalQA chain and question_set
predictions = qa.apply(question_set)

# Define the evaluation chain
eval_chain = QAEvalChain.from_llm(llm)

# Evaluate the ground truth against the answers that are returned
results = eval_chain.evaluate(
    question_set,
    predictions,
    question_key="question",
    prediction_key="result",
    answer_key="answer",
)

for i, q in enumerate(question_set):
    print(f"Question {i+1}: {q['question']}")
    print(f"Expected Answer: {q['answer']}")
    print(f"Model Prediction: {predictions[i]['result']}\n")

print(results)

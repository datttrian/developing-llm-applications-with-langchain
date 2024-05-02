from langchain.agents.agent_types import AgentType
from langchain.agents.initialize import initialize_agent
from langchain.agents.load_tools import load_tools
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.llms.base import OpenAI


model = ChatOpenAI()
prompt = ChatPromptTemplate.from_template(
    "You are a skilled poet. Write a haiku about the following topic: {topic}"
)

# Define the chain using LCEL
chain = prompt | model

# Invoke the chain with any topic
print(chain.invoke({"topic": "Large Language Models"}))


# Create the retriever and model
vectorstore = Chroma.from_texts(
    ["LangChain v0.1.0 was released on January 8, 2024."], embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
model = ChatOpenAI(temperature=0)

template = (
    """Answer the question based on the context:{context}. Question: {question}"""
)
prompt = ChatPromptTemplate.from_template(template)

# Create the chain and run it
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model  # type: ignore

chain.invoke("When was LangChain v0.1.0 released?")  # type: ignore


coding_prompt = PromptTemplate.from_template(
    """Write Python code to loop through the following list, printing each element: {list}"""
)
validate_prompt = PromptTemplate.from_template(
    """Consider the following Python code: {answer} If it doesn't use a list comprehension, update it to use one. If it does use a list comprehension, return the original code without explanation:"""
)

llm = ChatOpenAI()

# Create the sequential chain
chain = (
    {"answer": coding_prompt | llm | StrOutputParser()}
    | validate_prompt
    | llm
    | StrOutputParser()  # type: ignore
)

# Invoke the chain with the user's question
chain.invoke({"list": "[3, 1, 4, 1]"})


# Make ceo_response available for other chains
ceo_response = (
    ChatPromptTemplate.from_template(
        "You are a CEO. Describe the most lucrative consumer product addressing the following consumer need in one sentence: {input}."  # type: ignore
    )
    | ChatOpenAI()
    | {"ceo_response": RunnablePassthrough() | StrOutputParser()}
)

advisor_response = (
    ChatPromptTemplate.from_template(
        "You are a strategic adviser. Briefly map the outline and business plan for {ceo_response} in 3 key steps."
    )
    | ChatOpenAI()
    | StrOutputParser()
)

overall_response = (
    ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "CEO response:\n{ceo_response}\n\nAdvisor response:\n{advisor_response}",
            ),
            (
                "system",
                "Generate a final response including the CEO's response, the advisor response, and a summary of the business plan in one sentence.",
            ),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

# Create a chain to insert the outputs from the other chains into overall_response
business_idea_chain = (
    {"ceo_response": ceo_response, "advisor_response": advisor_response}  # type: ignore
    | overall_response
    | ChatOpenAI()
    | StrOutputParser()
)

print(
    business_idea_chain.invoke(
        {
            "input": "Typing on mobile touchscreens is slow.",
            "ceo_response": "",
            "advisor_response": "",
        }
    )
)


llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)  # type: ignore

# Define the tools
tools = load_tools(["llm-math"], llm=llm)

# Define the agent
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# Run the agent
agent.run("What is 10 multiplied by 50?")

import os

import openai
from dotenv import load_dotenv
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.evaluation.qa.eval_chain import QAEvalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set your API Key from OpenAI
load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

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

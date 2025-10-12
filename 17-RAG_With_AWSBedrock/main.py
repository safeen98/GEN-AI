import boto3
import streamlit as st
from langchain_aws import BedrockEmbeddings
from langchain_aws import BedrockLLM
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

client = boto3.client('bedrock-runtime')
embedding = BedrockEmbeddings(model_id='amazon.titan-embed-text-v2:0',client=client)
llm = BedrockLLM(model_id = 'qwen.qwen3-coder-30b-a3b-v1:0',client = client)

loader = PyPDFDirectoryLoader('/Users/mohdsafeenkhan/Desktop/Machine_Learning/GEN-AI/17-RAG_With_AWSBedrock/research_papers')
document = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 800,chunk_overlap = 500)
splitted_doc = splitter.split_documents(document)
vectorDB = FAISS.from_documents(splitted_doc,embedding=embedding)
retrival = vectorDB.as_retriever()

prompt = ChatPromptTemplate.from_template("""
You are a helpful chatBot used for question and answer task , provide the answer based on the following context only :
<context>
{context}
<context>
Question :{input}
CRITICAL RULES: 
    - DO NOT PROVIDE ANYTHINNG OUT OF THE CONTEXT
    - IF YOU DO NOT KNOW THE ANSWER JUST SAY IT IS OUT OF CONTEXT

""")

qa_chain = create_stuff_documents_chain(llm,prompt)
retrivalChain = create_retrieval_chain(retrival , qa_chain)

st.title('ChatBot using AWS Bedrock')

user_query = st.text_input('Ask the question based on the document loaded')

if user_query:
    response = retrivalChain.invoke({'input':user_query})
    st.success(response['output'])
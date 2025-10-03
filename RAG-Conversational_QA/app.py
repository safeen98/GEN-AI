import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import time

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='meta-llama/llama-4-maverick-17b-128e-instruct')

prompts = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question : {input}
    """
)

def create_vector_embeddings():
    if 'vectors' not in st.session_state: # we have used this to maintain the state of the application so that it remembers or has some histroy of the document 
        st.session_state.embedding = OllamaEmbeddings(model='embeddinggemma:300m')
        st.session_state.loader = DirectoryLoader('/Users/mohdsafeenkhan/Desktop/Machine_Learning/GEN-AI/RAG-Conversational_QA/research_papers',glob="*.pdf",loader_cls=PyPDFLoader) # Feeding the data
        st.session_state.doc = st.session_state.loader.load()
        st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
        st.session_state.splittedDoc = st.session_state.splitter.split_documents(st.session_state.doc)
        st.session_state.vectordb = FAISS.from_documents(st.session_state.splittedDoc ,st.session_state.embedding)

user_prompt = st.text_input('Enter your query from the research Paper')

if st.button('Create VecorDB'):
    create_vector_embeddings()
    st.write('VectorDB is ready')

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompts)
    retriever = st.session_state.vectordb.as_retriever()
    retrival_chain = create_retrieval_chain(retriever,document_chain)
    start =time.process_time()
    response = retrival_chain.invoke({
        'input':user_prompt
    })
    print(f"Response Time : {time.process_time()-start}")
    st.write(response['answer'])
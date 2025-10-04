import streamlit as st
import os
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever , create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import OllamaEmbeddings
import uuid
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

st.title("Conversational RAG with PDF upload and chat history")
st.write("Upload PDFs and chat with their content")

llm = ChatGroq(model= "meta-llama/llama-4-maverick-17b-128e-instruct")

session_id = st.text_input("Current Session id", value = str(uuid.uuid4()))

if "store" not in st.session_state:
    st.session_state.store = {}

uploadded_Files = st.file_uploader("Upload the PDF file",type="pdf",accept_multiple_files=True)

if uploadded_Files:
    document =[]
    for uploadded_File in uploadded_Files:
        tempFile = f"./temp.pdf"
        with open(tempFile,"wb") as file:
            file.write(uploadded_File.getvalue())
            file_name = uploadded_File.name
        loader = PyPDFLoader(tempFile)
        doc = loader.load()
        document.extend(doc)
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
    splitted_document = splitter.split_documents(document)
    embedder = OllamaEmbeddings(model="embeddinggemma:300m")
    vectordb = FAISS.from_documents(splitted_document,embedder)
    retriver = vectordb.as_retriever()

    contextualize_prompt = ("""
    Given the chat history and the latest user question
    which might reference context n the chat history
    formulate a standalone questio which can be understood
    withour the chat histroy.DO NOT ANSWER THE QUESTION,
    just reformulate it if nedded and otherwise return it as is
    """)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system",contextualize_prompt),
        MessagesPlaceholder("chat_history"),
        ("user","{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm,retriver,chat_prompt)

    system_prompt = ("""
    You are an assistant for question - answer task
    Use the following piece of retrieved context to answer the question.
    If you do not know the answer,say that you don not know.
    Use three sentense maximum and keep the answer concise
    {context}
    """)

    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user","{input}")
    ])

    QAChain = create_stuff_documents_chain(llm,prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever,QAChain)

    def get_session(session_id:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session,
    input_messages_key="input",
    history_messages_key="chat_history",
)

    user_input = st.text_input("Ask your question")
    if user_input:
        response = conversational_rag_chain.invoke({"input":user_input},config={
        "configurable":{"session_id":session_id}
        })
        st.write(response['answer'])
        st.write(response)
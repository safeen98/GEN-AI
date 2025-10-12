import streamlit as st
import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableWithMessageHistory
load_dotenv()

llm = ChatOllama(model='JARVIS')

chat_history = {}
def getSession(session_id:str)->BaseChatMessageHistory:
    if session_id not in chat_history:
        chat_history[session_id] = ChatMessageHistory()
    return chat_history[session_id]

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder('chat_history'),
    ('user','{input}')
])
str_parser = StrOutputParser()
llm_chain = prompt |llm | str_parser

history_chain = RunnableWithMessageHistory(llm_chain,getSession,input_messages_key='input',history_messages_key='chat_history')

st.title('CodeHelper Using Custom Model')
user_query = st.text_input('Ask any coding related question')

if user_query:
    config = {'configurable':{
        'session_id' : 'session1'
    }}

    response = history_chain.invoke({'input':user_query},config=config)
    st.write(response)

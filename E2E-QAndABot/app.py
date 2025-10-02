import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

#langsmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACKING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'QAndA ChatBot'

#prompt Template

prompt = ChatPromptTemplate.from_messages(
    [
        ('system','Your are a helpful bot , provide the answer to the below question a very precise manner, also explain a manner how a gut from Uttar pradesh Talks, provide answer in Hinglish'),
        ('human','{question}')
    ]
)

def generate_response(question,api_key):
    os.environ['GROQ_API_KEY'] = api_key
    strParser = StrOutputParser()
    llm = ChatGroq(model='llama-3.3-70b-versatile')
    chain = prompt | llm | strParser
    return chain.invoke({
        'question':question
    })

# Streamlit setup

st.title('Q&A Bot using LLM')

st.sidebar.title('Setting')
api_key=st.sidebar.text_input('Enter you API Key',type='password')

st.write('Go ahead and ask any question')

user_input = st.text_input('You:')
if user_input:
    response = generate_response(user_input,api_key)
    st.write(response)
else:
    st.write('Please provide a query')
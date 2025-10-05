import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

st.title('Langchain : Chat with SQL DB')

LOCALDB = 'USE_LOCALDB'

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='llama-3.3-70b-versatile',streaming=True)

@st.cache_resource(ttl='2h')
def config_db():
    dbFilePath = (Path(__file__).parent/'student.db').absolute()
    print(dbFilePath)
    creator = lambda :sqlite3.connect(f"file:{dbFilePath}?mode=ro",uri=True)
    return SQLDatabase(create_engine('sqlite:///',creator=creator))

database = config_db()

toolkit = SQLDatabaseToolkit(db=database,llm=llm)

agent = create_sql_agent(llm=llm,toolkit=toolkit,agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,verbose=True)

if 'messages' not in st.session_state:
    st.session_state['messages'] = [{'role':'assistant','content':'How can I help'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
user_query = st.chat_input(placeholder='Ask anything from the database')

if user_query:
    st.session_state.messages.append({'role':'user','content':user_query})
    st.chat_message('user').write(user_query)

    with st.chat_message('assistant'):
        streamlit_callbacks = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query,callbacks=[streamlit_callbacks])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
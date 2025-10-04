import streamlit as st
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools import WikipediaQueryRun , ArxivQueryRun , DuckDuckGoSearchRun
from langchain.agents import  initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name='SearchEngine')
tools=[wiki,arxiv,search]

st.title('Langchain , Chat with search')

if 'messages' not in st.session_state:
    st.session_state['messages']=[{
        'role':'assistant','content':'Hi , I am a chatbot'
    }]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
if prompt:=st.chat_input(placeholder='What is machine learning ?'):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('user').write(prompt)
    llm = ChatGroq(model = 'meta-llama/llama-4-maverick-17b-128e-instruct',streaming=True)
    search_agent = initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)

    with st.chat_message('assistant'):
        st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages , callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
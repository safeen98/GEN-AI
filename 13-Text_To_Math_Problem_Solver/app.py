import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType,AgentExecutor,initialize_agent,create_react_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool
from langchain.callbacks import StreamlitCallbackHandler
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='llama-3.3-70b-versatile')

st.title('Text to Math Solver')

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)
calculation_tool = PythonREPLTool()
tools = [wiki_tool,calculation_tool]
prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT:
- The Action must be EXACTLY one of: {tool_names}
- DO NOT add any extra text in the Action line, just the tool name

Begin!

CRITICAL RULES:
- After you have received an Observation with the answer, YOU MUST IMMEDIATELY provide the Final Answer
- DO NOT REPEAT THE SAME Action MULTIPLE TIMES

Question: {input}
Thought:{agent_scratchpad}
""")
agent = create_react_agent(llm=llm,tools=tools,prompt=prompt)
agent_executor = AgentExecutor(agent=agent,tools=tools,verbose=True,handle_parsing_errors=True)
if 'messages' not in st.session_state:
    st.session_state['messages']=[{'role':'assistant','content':'Hii how can I help you today'}]
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])
user_query = st.text_input('Please provide a Math problem')
answer_button = st.button('Answer')
if answer_button:
    if not user_query:
        st.warning('Please provide a question to continue')
    else:
        st.session_state.messages.append({'role':'user','content':user_query})
        st.chat_message('user').write(user_query)
        with st.spinner('Calculating...'):
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
            response = agent_executor.invoke({'input':user_query},{'callbacks':[st_cb]})
            st.session_state.messages.append({'role':'assistant','content':response})
            st.chat_message('assistant').write(response)
            st.write(response['output'])

    



import validators
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model='llama-3.3-70b-versatile')

chunk_prompt = ChatPromptTemplate.from_template("""
Please summarize the below content:
content : {text}
""")

final_prompt = ChatPromptTemplate.from_template("""
please provide the summary for the already summarized text:
summary :{text}

""")

st.title('SUmmarize text from Youtube Video or Website')

url = st.text_input('URL',label_visibility='collapsed')

if st.button('Summarize the content'):
    if not url.strip():
        st.error('Please provide a url to summarize')
    elif not validators.url(url):
        st.error('Please provide a valid url')
    else:
        try:
            with st.spinner('Waiting...'):
                if 'youtube.com' or 'youtu.be' in url:
                    print(url)
                    loader = YoutubeLoader.from_youtube_url(url,add_video_info=False)
                else:
                    loader = UnstructuredURLLoader(urls=[url],ssl_verify=False)
                docs = loader.load()
                print(docs)
                
                chain = load_summarize_chain(llm,chain_type='map_reduce',map_prompt=chunk_prompt,combine_prompt=final_prompt,verbose=True)
                output_summary = chain.invoke(docs)
                st.success(output_summary['output_text'])
        except Exception as e:
            st.exception(f"Caught Exception {e}")
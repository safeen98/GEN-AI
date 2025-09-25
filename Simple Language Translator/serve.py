from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from langserve import add_routes
from dotenv import load_dotenv
#from langserve.validation import chainBatchRequest
#chainBatchRequest.model_rebuild()
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

#Defining our model
model = ChatGroq(model = "llama-3.3-70b-versatile")

# Defining the prompt
system_tempelate = "Translate the following to {language}"

prompt_template = ChatPromptTemplate.from_messages([
    ("system",system_tempelate),
    ("user","{text}")
])

#Defining the parser

parser = StrOutputParser()

#Chaining all these

chain = prompt_template|model|parser

#App definition

app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "A simple API server using langchain runable interfaces"
)

add_routes(
    app,
    chain,
    path = '/chain'
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host="127.0.0.1",port=8000)



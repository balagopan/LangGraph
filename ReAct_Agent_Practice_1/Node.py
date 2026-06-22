
from typing import TypedDict, Annotated, List, Union
from langgraph.graph.message import add_messages
from langchain.messages import AIMessage
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

def agent_state(Typeddict):
    input:str
    intermediate_ste:Annotated[List, add_messages]
    output:Union[AIMessage,None]

prompt_template=ChatPromptTemplate.from_messages([
    ("system","You are an AI assistant, who uses as less tokens as possible to help the user"),
    ("human","{input}")
])

llm = ChatGroq(model="groq/compound")

def responder_node(state:agent_state)->AIMessage:
    result=llm.invoke({"input":state})
    return {"output":result}

def tool_node(state:agent_state):
    



from typing import TypedDict, Annotated, List, Union
from langchain.messages import AIMessage
from langgraph.graph import add_messages, END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import datetime
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    input:str
    intermediate_step:Annotated[List, add_messages]
    output:Union[AIMessage,None]


@tool
def get_system_time(format:str="%Y:%m:%d %H-%m-%S"):
    """Returns the system time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time=current_time.strftime(format)
    return formatted_time

search_tool=TavilySearch(search_depth="basic")

tools=[get_system_time,search_tool]

llm=llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm_with_tools=llm.bind_tools(tools=tools)

prompt_template_main=ChatPromptTemplate.from_messages([
    ("system","You are an AI Assistant, You should use tools find answers to questions you dont know"
    "The information you have are follows:"
    "{intermediate_step}"),
    ("human","{input}")
])

responder_chain=prompt_template_main | llm_with_tools

def responder_node(state:AgentState):
    result=responder_chain.invoke(state)
    return {"output":result}

def tool_node(state:AgentState):
    result=[]
    lastAIMessage=state["output"]
    for i,tool_list in enumerate(lastAIMessage.tool_calls):
        tool_args=tool_list["args"]
        tool_name=tool_list["name"]
        tool_to_run=next((t for t in tools if tool_name==t.name),None)

        if tool_to_run:
            tool_result=tool_to_run.invoke(tool_args)
        else:
            tool_result=f"Use only {tools} as tools"
        result.append((tool_name,tool_result))

    return {"intermediate_step":result}

def should_continue(state:AgentState):
    lastAIMessage=state["output"]
    if isinstance(lastAIMessage,AIMessage) and hasattr(lastAIMessage,"tool_calls") and len(lastAIMessage.tool_calls>0):
        return "tool_executer"
    else:
        return END
    
graph=StateGraph(AgentState)

graph.add_node("responder",responder_node)
graph.add_node("tool_executor",tool_node)
graph.set_entry_point("responder")
graph.add_conditional_edges("responder",should_continue)
graph.add_edge("tool_executor","responder")

app=graph.compile()

prompt_template_exit=ChatPromptTemplate.from_messages([
    ("system","You are an AI chatbot user message interpreter who converts user messsage into hardcoded text"
     "You act as a bouncer to let the user exit or continue the conversation"
    "Your only job is o interpret the user message and find out if the user wishes to exit from the chat bot, If the user wisher to exit You"
    "exactly return the string 'EXIT' else return the user message as it is"),
    ("human","{input}")
])

bouncer_llm= prompt_template_exit | llm

while True:
    user_input=input("User :")
    bouncer_result=bouncer_llm.invoke({"input":user_input})
    if bouncer_result=="EXIT":
        print("AI :Have a great day!")
        break

    answer=app.invoke({
        "input":user_input,
        "intermediate_step":[],
        "output":None
    })

    print("AI :{answer.output.content}")


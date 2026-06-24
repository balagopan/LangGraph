
from typing import TypedDict, Annotated, List, Union
from langchain.messages import AIMessage
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import datetime
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
# from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_groq import ChatGroq
import operator
from groq import BadRequestError
from typing import Optional

sqlite_connect=sqlite3.connect("Sqlite_database.sqlite", check_same_thread=False)

memory=SqliteSaver(sqlite_connect)

load_dotenv()

class AgentState(TypedDict):
    input:str
    chat_history:Annotated[List, operator.add]
    errors:Union[List,None]
    tool_results:Annotated[List, operator.add]
    output:Union[AIMessage,None]


@tool
def get_system_time(format: Optional[str] = None):
    """Returns the system time in the specified format"""
    if format == None:
        format="%Y:%m:%d %H-%m-%S"
    current_time = datetime.datetime.now()
    formatted_time=current_time.strftime(format)
    return formatted_time

search_tool=TavilySearch(search_depth="basic")

tools=[get_system_time,search_tool]

# llm=llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
# llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
# llm = ChatGroq(model="groq/compound")

llm_with_tools=llm.bind_tools(tools=tools)


prompt_template_main=ChatPromptTemplate.from_messages([
    ("system","You are an AI Agent built using langchain. "
    "Chat history: \n"
    "{chat_history}\n"
    "Tool results:\n "
    "{tool_results}\n"
    "For the current process the following error was encountered by the llm, address this: \n"
    "{errors}"),
    ("human","{input}"),
    ("system","If the users question can be answered using tool results above, do not use the tools.")
])

responder_chain=prompt_template_main | llm_with_tools

def responder_node(state:AgentState):
    max_tries=3
    current_state=state
    for attempt in range(max_tries):        
        try:
            result=responder_chain.invoke(current_state)
            if isinstance(result,AIMessage) and hasattr(result,"tool_calls") and len(result.tool_calls)>0:
                return {"output":result}
            else:
                return {"output":result.content[0]["text"],"chat_history":[current_state["input"],result.content[0]["text"]]}
        except Exception as e:
            last_error = e
            print(f"Formatting error on attempt {attempt + 1}: {e}")

            error_msg=([
                "system",f"Your last generation failed with an API formatting error: {str(e)}."
                 ])
            current_state["errors"]=[error_msg]
        
    result=f"Formatting error still persists on attempt {attempt}: {last_error}"
    return {"errors": current_state.get("errors", []) + [result]}

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
        
    return {"tool_results":result}

def should_continue(state:AgentState):
    lastAIMessage=state["output"]
    if isinstance(lastAIMessage,AIMessage) and hasattr(lastAIMessage,"tool_calls") and len(lastAIMessage.tool_calls)>0:
        return "tool_executor"
    else:

        return END
    
graph=StateGraph(AgentState)

graph.add_node("responder",responder_node)
graph.add_node("tool_executor",tool_node)
graph.set_entry_point("responder")
graph.add_conditional_edges("responder",should_continue)
graph.add_edge("tool_executor","responder")

app=graph.compile(memory)

config={"configurable":{
    "thread_id":2
}}

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
    if "EXIT" in bouncer_result.content:
        print("AI :Have a great day!")
        break

    answer=app.invoke({
        "input":user_input,
        "chat_history":[],
        "errors":[],
        "tool_results":[],
        "output":None
    },config=config)

    print(f"AI :{answer["output"]}")

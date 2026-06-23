
from typing import TypedDict, Annotated, List, Union
from langchain.messages import AIMessage
from langgraph.graph import END, StateGraph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
import datetime
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
import operator
from groq import BadRequestError

memory=MemorySaver()

load_dotenv()

class AgentState(TypedDict):
    input:str
    chat_history:Annotated[List, operator.add]
    intermediate_step:Annotated[List, operator.add]
    output:Union[AIMessage,None]


@tool
def get_system_time(format:str="%Y:%m:%d %H-%m-%S"):
    """Returns the system time in the specified format"""
    current_time = datetime.datetime.now()
    formatted_time=current_time.strftime(format)
    return formatted_time

search_tool=TavilySearch(search_depth="basic")

tools=[get_system_time,search_tool]

# llm=llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
# llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0)
# llm = ChatGroq(model="groq/compound")

llm_with_tools=llm.bind_tools(tools=tools)


prompt_template_main=ChatPromptTemplate.from_messages([
    ("system","You are an AI Assistant"
    "The chat history you had are as follows"
    "{chat_history}"
    "The information you have by running tools are follows:"
    "{intermediate_step}"),
    ("human","{input}"),
    "If you cant find the answer to  the user query from the tools information and chat history, use tools"
])

responder_chain=prompt_template_main | llm_with_tools

def responder_node(state:AgentState):
    max_tries=3
    current_state=state
    for attempt in range(max_tries):
        try:
            result=responder_chain.invoke(current_state)
            return {"output":result,"intermediate_step": current_state.get("intermediate_step", []) + [result.content],"chat_history":[current_state["input"],result.content]}
        except Exception as e:
            last_error = e
            print(f"Formatting error on attempt {attempt + 1}: {e}")

            error_msg=([
                "system",f"Your last generation failed with an API formatting error: {str(e)}."
                 ])
            current_state["intermediate_step"]+[error_msg]
        
    result=f"Formatting error still persists on attempt {attempt}: {last_error}"
    return {"output":result,"intermediate_step": current_state.get("intermediate_step", []) + [result]}

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
    "thread_id":1
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
        "intermediate_step":[],
        "output":None
    },config=config)

    print(f"AI :{answer['output'].content}")


from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
import datetime
from langchain_tavily import TavilySearch
from langchainhub import Client
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.prebuilt import ToolNode
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
# from openai import OpenAI

load_dotenv()


# llm = ChatGroq(model="llama-3.3-70b-versatile")
# llm = ChatGroq(model="llama-3.1-8b-instant")
llm = ChatGroq(model="groq/compound")
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

search_tool=TavilySearch(search_depth="basic")

@tool
def get_system_time():
    """Returns the current system date and time in the specified format"""

    # format:str="%Y-%m-%d %H:%M:%S"
    current_time=datetime.datetime.now()
    formatted_time=current_time.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_time

tools=[search_tool, get_system_time]

# agent=create_agent(model=llm, tools=tools)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an advanced AI assistant."
    "Here is the data you have gathered so far from your tools:{intermediatestep}"
    "CRITICAL INSTRUCTION:"
    "Only use tools when you dont have the required information, if the data above contains the required information, DO NOT use any more tools. Provide the final answer in plain text."
    "If you do not know an anser always use tools, never guess or hallucinate"
    "If the user is asking multiple things do only one thinng"),
    ("human","{input}")
])

# llm_with_tools=llm.bind_tools(tools=tools, parallel_tool_calls=False)
# llm_with_tools=llm.bind_tools(tools=tools)
llm_with_tools=llm


reason_node=prompt_template | llm_with_tools

tool_node=ToolNode(tools)

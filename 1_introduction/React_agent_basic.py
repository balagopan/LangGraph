from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
# from langchainhub import Client


load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-2.5-flash"
)

system_instructions = (
    "You are a humorous and helpful AI assistant. "
    "If a user asks for current events or real-time data, ALWAYS use your search tool to find the facts first. "
    "Once you have the facts, complete the user's creative request based on that data."
)

# hub_client = Client()

# prompt_template=hub_client.pull("hwchase17/react")

search_tool=TavilySearch(search_depth="basic")

tools=[search_tool]

agent = create_agent(model=llm,
                     tools=tools,
                     system_prompt=system_instructions)

query="Give me a funny tweet about the whether condition of malaysia today?"

# result=agent.invoke({"messages":query})

# print(result["messages"][-1].content[0]["text"])

for step in agent.stream({"messages": [("user", query)]}):
    
    # If the agent node is thinking or deciding to use a tool
    if "model" in step:
        ai_message = step["model"]["messages"][-1]

        text_content=ai_message.content

        if text_content and ai_message.tool_calls:
            print(f"🤔 THOUGHT: {text_content}")
        
        # Check if the agent decided to call a tool
        if ai_message.tool_calls:
            for tool in ai_message.tool_calls:
                print(f"🛠️  ACTION: Calling tool '{tool['name']}' with arguments: {tool['args']}")
                
        if not ai_message.tool_calls:
            print("\n✅ FINAL TWEET:")
            print(text_content[0].text)
    
    # If the tool node is returning search results
    elif "tools" in step:
        tool_message = step["tools"]["messages"][-1]
        print(f"👀 OBSERVATION: {tool_message.content[:200]}... [truncated for readability]")


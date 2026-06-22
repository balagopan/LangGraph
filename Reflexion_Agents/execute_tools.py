from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from typing import List
from langchain_community.tools import tavily_search
import json


def execute_tools(state:List[BaseMessage])->List[BaseMessage]:
    last_ai_message: AIMessage =state[-1]

    if not hasattr(last_ai_message,"tool_calls") or last_ai_message.tool_calls:
        return[]
    
    tool_messages=[]

    for tool_calls in last_ai_message.tool_calls:
        if tool_calls["name"] in ['AnswerQuestion','ReflectAnswer']:
            call_id=tool_calls["id"]
            search_quaries = tool_calls[0]["args"].get("search_quaries", [])

            quary_result={}
            for quary in search_quaries:
                result=tavily_search.invoke(quary)
                quary_result[quary]=result

            tool_messages.append(ToolMessage(
                content=json.dumps(quary_result),
                tool_call_id=call_id
            ))
            
    return tool_messages
from langgraph.graph import StateGraph
from Agent_state import agent_state
from nodes import agent_node, tool_node

flow=StateGraph(agent_state)

def should_continue():
    pass

flow.add_node("agent",agent_node)
flow.add_node("tool",tool_node)
flow.add_edge("tool","agent")
flow.add_conditional_edges("agent",should_continue)
flow.set_entry_point("agent")

agent=flow.compile()

agent.invoke("When was the last SpaceX rocket launched?")

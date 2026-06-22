from Agent_state import agent_state
from react_agent_runnable import reason_node

def agent_node(state:agent_state):
    result=reason_node.invoke(state)
    return {"output": result}


    
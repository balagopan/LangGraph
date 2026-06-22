from typing import TypedDict, List
from langgraph.graph import END, StateGraph

class Simplestate(TypedDict):
    count:int
    total_sum:int
    history:List

def incriment(state):
    new_count=state["count"]+1
    return{
        "count":new_count,
        "total_sum":state["total_sum"]+new_count,
        "history":state["history"]+[new_count]
    }

def should_continue(state):
    if state["count"]>=5:
        return END
    else:
        return "incriment"
    
graph=StateGraph(Simplestate)

graph.add_node("incriment",incriment)
graph.set_entry_point("incriment")
graph.add_conditional_edges("incriment", should_continue)

app=graph.compile()

result=app.invoke(
    {"count":0,
     "total_sum":0,
     "history":[]}
)

print(result)

from dotenv import load_dotenv
from langgraph.graph import END, MessageGraph
from chains import generation_chain, reflection_chain
from langchain_core.messages import HumanMessage, BaseMessage

load_dotenv()

graph=MessageGraph()

def generate_node(state):
    return generation_chain.invoke({"message":state})

def reflection_node(state):
    response=reflection_chain.invoke({"message":state})
    return HumanMessage(content=response.content)

graph.add_node("generate",generate_node)
graph.add_node("reflect",reflection_node)

graph.set_entry_point("generate")

def should_continue(state):
    # print(f"\n\n{state}")
    if len(state)>4:
        return END
    elif(state[-1].content=="I dont have any improvements to suggest"):
        return END
    else:
        return "reflect"

graph.add_conditional_edges("generate",should_continue,
                            {
        END: END,            # If function returns END, route to the END node
        "reflect": "reflect" # If function returns "reflect", route to the reflect node
    })
graph.add_edge("reflect","generate")

app=graph.compile()

# print(app.get_graph().draw_mermaid())
# app.get_graph().print_ascii()

result=app.invoke(HumanMessage(content="AI taking over human jobs"))

for i in result:
    print(f"\n\n{result[i].content}")

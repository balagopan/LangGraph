from typing import TypedDict, Annotated, List, Union
from langgraph.graph.message import add_messages
import operator

class agent_state(TypedDict):
    input:str
    intermediatestep:Annotated[List[tuple[str,str]], operator.add]
    output:Union[str,None]
    
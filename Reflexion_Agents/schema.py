from pydantic import BaseModel, Field
from typing import List


class Reflection(BaseModel):
    missing:str=Field(description="critique of what is missing")
    superfluous:str=Field(description="critique of what is superfluous")

class AnswerQuestion(BaseModel):
    """The format template for the answer"""
    answer:str=Field(description="250 words answer to the question")
    reflection:Reflection=Field(description="The reflection on the answer")
    search_quaries:List[str]=Field(description="1-3 search quaries to researching improvements to address the critique of the current answer")

class ReflectAnswer(AnswerQuestion):
    citations:str=Field(description="Citations of where you found the source of your answers")
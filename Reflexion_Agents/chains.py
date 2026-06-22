
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import datetime
# from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from schema import AnswerQuestion, ReflectAnswer
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser

load_dotenv()
# llm=llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

structured_llm_responder=llm.with_structured_output(AnswerQuestion)
structured_llm_revisor=llm.with_structured_output(ReflectAnswer)

actor_prompt_template=ChatPromptTemplate.from_messages(
    [
        ("system",
         """You are an expert AI researcher.
         CUrrent time={time}
         
         1.{instruction}
         2.Reflect and critique on the above statement.
         3.After reflection and critiquing, list 1-3 search quaries to 
         research improvements, Do not include them inside the reflection""",),
         MessagesPlaceholder(variable_name='messages'),
         ("system","Answer the user question above in the required format"),
    ]
).partial(time=lambda: datetime.datetime.now().isoformat(),)

revisor_prompt="""Provide a 250 words detailed revised answer addressing all the critisisms mentioned in
the previous one and improving the answer with the contens from the search results.
-You must address the critisms mentioned in the missing and superfluous.
-The answer must contain citations, supporting all the information in the answer.
-The word limit should be strictly less that on equal to 250
"""

responder_prompt_template=actor_prompt_template.partial(instruction="Provide a 250 words detatiled answer")
revisor_prompt_template=actor_prompt_template.partial(instruction=revisor_prompt)

validator_responder = PydanticToolsParser(tools=[AnswerQuestion])
validator_revisor = PydanticToolsParser(tools=[ReflectAnswer])

responder_chain=responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion') 
revisor_chain=revisor_prompt_template |llm.bind_tools(tools=[ReflectAnswer], tool_choice='ReflectAnswer') 

response=responder_chain.invoke({"messages":[HumanMessage(
    content="Write a blog post on how startups can leverage AI to increase their growth rate")]})

print(response)

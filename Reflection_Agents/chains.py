from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

generation_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a techie twitter influencer assistant tasked with writing excellent twitter posts "
            "Generate the best twitter post possible for the user request"
            "If the user provides ant critique, respond with a revised version of the given tweet"
        ),
        MessagesPlaceholder(variable_name="message"),
    ]
)

reflection_prompt=ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a techie twitter influencer assistant tasked with grading twitter posts"
            "Generate critique and recomendations to make the tweet better"
            "Provide a detailed review highlighting the areas to improve and how it can be improved. "
            "If you do not have any suggestions, say EXACTLY: 'I dont have any improvements to suggest'."
        ),
        MessagesPlaceholder(variable_name="message"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

generation_chain=generation_prompt | llm
reflection_chain= reflection_prompt | llm

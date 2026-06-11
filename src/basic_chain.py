from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from config import GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=GOOGLE_API_KEY,
)

prompt = ChatPromptTemplate.from_template(
    """
    You are a climate science tutor.

    Explain the following topic clearly in 3 sentences:
    {topic}
    """
)

chain = prompt | llm 

response = chain.invoke({
    "topic": "PM2.5 and public health"
})

print(response.text)
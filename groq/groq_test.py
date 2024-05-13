from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# chat = ChatGroq(temperature=0, model_name="llama3-70b-8192")

chat = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

system = "You are a helpful assistant."
human = "{text}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

chain = prompt | chat
response = chain.invoke({"text": "Explain the importance of low latency LLMs."})
print(response)

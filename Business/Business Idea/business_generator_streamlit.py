import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")

# Define the chain and LLM first
llm = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages([
    ("system", '''
    You are creative senior business consultant. You will be asked to provide a business idea based on a topic. 
    Please provide a creative business idea based on the topic and step by step guide to implement it.
    '''),
    ("user", "Can you give me business idea about {topic}")
])
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Create gradio interface function with topic as input
def generate_business_idea(topic):
    output = chain.invoke({"topic": topic})
    return output

# Define Streamlit interface
st.title("Business Idea Generator 💵")
st.write("Enter a topic, and I'll provide a creative business idea!")

topic = st.text_input("Topic")
if st.button('Generate'):
    if topic != "":
        idea = generate_business_idea(topic)
        st.write(idea)
    else:
        st.write("Please enter a topic.")

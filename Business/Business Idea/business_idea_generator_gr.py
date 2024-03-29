import gradio as gr
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

# Define gradio interface
demo = gr.Interface(
    fn=generate_business_idea,
    inputs="text",
    outputs="text",
    title="Business Idea Generator",
    description="Enter a topic, and I'll provide a creative business idea!",
)

if __name__ == "__main__":
    demo.launch()

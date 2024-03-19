from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

openai_api_key = os.environ.get("OPENAI_API_KEY")


llm = ChatOpenAI()
# Examples of a pretend task of creating antonyms.
examples = [
    {"input": "The movie was fantastic!", "output": "Positive"},
    {"input": "I didn't enjoy the food at all", "output": "Negative"},
    {"input": "Amazing vacation, I had a great time!", "output": "Positive"},
    {"input": "She looks upset and angry.", "output": "Negative"},
]

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
example_selector = LengthBasedExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    max_length=25,
)
dynamic_prompt = FewShotPromptTemplate(
    # We provide an ExampleSelector instead of examples.
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the sentiment positive or negative of every input",
    suffix="Input: {sentence}\nOutput:",
    input_variables=["sentence"],
)
output_parser = StrOutputParser()
chain = dynamic_prompt | llm | output_parser

# Create gradio interface function with topic as input
def sentiment_analysis(sentence):
    output = chain.invoke({"sentence": sentence})
    return output

# Define gradio interface
demo = gr.Interface(
    fn=sentiment_analysis,
    inputs="text",
    outputs="text",
    title="Sentiment Analysis",
    description="Enter a sentence, and I'll provide a sentiment",
)

if __name__ == "__main__":
    demo.launch()
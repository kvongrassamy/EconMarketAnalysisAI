from langchain_tavily import TavilySearch
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain.tools import StructuredTool
import operator
from typing import Annotated, List, TypedDict
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from typing import Literal
from langchain.chains.combine_documents.reduce import (
    acollapse_docs,
    split_list_of_docs,
)
from datetime import datetime
import os
import json
from langchain.agents import tool
from dotenv import load_dotenv
import asyncio
import time


load_dotenv()

os.environ["TAVILY_API_KEY"] = os.environ.get("TAVILY_API_KEY")

tavily_search_tool = TavilySearch(
    max_result = 6,
    topic = 'news',
    include_answer = True,
    search_depth = 'advanced'
)

@tool
def format_and_store(word: str) -> str:
    """
    Format the message before it is stored into a txt file
    """
    word = word.format()
    #print(word)
    with open("data/output.txt", "a") as file:
        file.writelines(f"{word} \n\n")

    return word

@tool
def directory_reader(word: str) -> str:
    """
    This tool will retrieve metadata about the current state of the economy which is provided by the Market Researcher

    """

    with open('data/output.txt', 'r') as file:
        data = file.read().rstrip()

    return data

@tool
def pdf_reader(word: str) -> str:
    """
    This tool is to provide suggestion, recommendations, or information on the current state of the economy provided in output.txt
    """

    pdf = "EconText/DLS1.pdf"
    loader = PyPDFLoader(pdf).load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(loader, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.7})
    docs = retriever.invoke(word)

    metadata_list = [doc for doc in docs]
    # Once you have the metadata_list readt the output.txt file to see if anything is realted to the PDF File
    return metadata_list



llm = init_chat_model("gpt-4o-mini", model_provider="openai")

map_template = "Write a concise summary of the following: {context}."

reduce_template = """
The following is a set of summaries:
{docs}
Take these and distill it into a final, consolidated summary
of the main themes.
"""

map_prompt = ChatPromptTemplate([("human", map_template)])
reduce_prompt = ChatPromptTemplate([("human", reduce_template)])

map_chain = map_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()

class SummaryState(TypedDict):
    content: str

class OverallState(TypedDict):
    # Notice here we use the operator.add
    # This is because we want combine all the summaries we generate
    # from individual nodes back into one list - this is essentially
    # the "reduce" part
    contents: List[str]
    summaries: Annotated[list, operator.add]
    final_summary: str

# Here we generate a summary, given a document
async def generate_summary(state: SummaryState):
    response = await map_chain.ainvoke(state["content"])
    return {"summaries": [response]}


def length_function(documents: List[Document]) -> int:
    """Get number of tokens for input contents."""
    return sum(llm.get_num_tokens(doc.page_content) for doc in documents)


def map_summaries(state: OverallState):
    # We will return a list of `Send` objects
    # Each `Send` object consists of the name of a node in the graph
    # as well as the state to send to that node
    return [
        Send("generate_summary", {"content": content}) for content in state["contents"]
    ]

token_max = 1000


class OverallState(TypedDict):
    contents: List[str]
    summaries: Annotated[list, operator.add]
    collapsed_summaries: List[Document]  # add key for collapsed summaries
    final_summary: str


# Add node to store summaries for collapsing
def collect_summaries(state: OverallState):
    return {
        "collapsed_summaries": [Document(summary) for summary in state["summaries"]]
    }


# Modify final summary to read off collapsed summaries
async def generate_final_summary(state: OverallState):
    response = await reduce_chain.ainvoke(state["collapsed_summaries"])
    return {"final_summary": response}

graph = StateGraph(OverallState)
graph.add_node("generate_summary", generate_summary)  # same as before
graph.add_node("collect_summaries", collect_summaries)
graph.add_node("generate_final_summary", generate_final_summary)

# Add node to collapse summaries
async def collapse_summaries(state: OverallState):
    doc_lists = split_list_of_docs(
        state["collapsed_summaries"], length_function, token_max
    )
    results = []
    for doc_list in doc_lists:
        results.append(await acollapse_docs(doc_list, reduce_chain.ainvoke))

    return {"collapsed_summaries": results}


graph.add_node("collapse_summaries", collapse_summaries)


def should_collapse(
    state: OverallState,
) -> Literal["collapse_summaries", "generate_final_summary"]:
    num_tokens = length_function(state["collapsed_summaries"])
    if num_tokens > token_max:
        return "collapse_summaries"
    else:
        return "generate_final_summary"


graph.add_conditional_edges(START, map_summaries, ["generate_summary"])
graph.add_edge("generate_summary", "collect_summaries")
graph.add_conditional_edges("collect_summaries", should_collapse)
graph.add_conditional_edges("collapse_summaries", should_collapse)
graph.add_edge("generate_final_summary", END)
app = graph.compile()


async def textbook_review(word: str, path_list: List) -> str:
    """
    This tool is to provide summary of textbooks.  Please attach all paths in a List to the path_list argument
    Remove any \\ in the name
    """
    summaries = []
    async for path in path_list:
        documents = PyPDFLoader(path).load()

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        split_docs = text_splitter.split_documents(documents)
        async for step in app.astream(
            {"contents": [doc.page_content for doc in split_docs]},
            {"recursion_limit": 12}):
            if 'generate_final_summary' not in list(step.keys()):
                continue
            else:
                summaries.append(step['generate_final_summary']['final_summary'])

    summaries = "  ".join(summaries)
    return summaries

textbook_reader = StructuredTool.from_function( func=textbook_review)
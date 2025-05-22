import openai
from typing import Literal
from typing_extensions import TypedDict

from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langsmith import wrappers

from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from prompts import MARKET_RESEARCH_PROMPTS
from tools import tavily_search_tool, format_and_store, directory_reader, pdf_reader, textbook_reader
import os




members = ["economist_agent", "evaluator_agent"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=4096)
#llm = ChatOpenAI(temperature=0, model="gpt-4o-2024-11-20", max_tokens=4096)
#llm = ChatAnthropic(model="claude-3-5-sonnet-latest")


class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    response = llm.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})


def __init_marketresearch__():
    marketresearch_agent = create_react_agent(
        llm, tools=[tavily_search_tool, format_and_store], 
        state_modifier="""
    Research economic issues related to education, the labor force, international trade, and other topics
    Advise businesses, governments, and individuals on problems related to economic topics
    Present research that is related to articles you review with tavily_search_tool tool
    You will analyze topics related to the production, distribution, and use (consumption) of goods and services. They work in or across a variety of fields, such as business, health, and the environment. For example, some economists study the cost of products, healthcare, or energy, while others examine employment levels and trends, business cycles, inflation, or interest rates.
    You will study historical trends and make forecasts, using software to analyze data.
    You will review tavily_search_tool results about the economy, including employment, prices, productivity, and wages. For example, the tavily_search_tool tool will have something related to economics, provide in depth information on that.
    You will also review for international organizations, research firms, and consulting firms. They may present their findings to a variety of audiences or publish their analyses and forecasts in newspapers, journals, or other media.
    """
    )

    if os.path.exists('data/output.txt'):
        os.remove('data/output.txt')

    for message in MARKET_RESEARCH_PROMPTS:
        prompt = MARKET_RESEARCH_PROMPTS[message]
        for s in marketresearch_agent.stream(
            {"messages": [("user", f"""{prompt}.  Please use current news articles to 
                        conclude the current state of the economy and write your final messages to output.txt using format_and_store tool
                        """)]}, subgraphs=True
            ):
            print(s)
            print("----")




economist_agent = create_react_agent(llm, tools=[directory_reader, pdf_reader],
                                 state_modifier="""
First you will need to read the output.txt file which you can use the directory_reader tool.
Next, review the PDF textbook using the pdf_reader tool to provide additional information on topics provided by directory_reader.
                                                    """)


def economist_agent_node(state: State) -> Command[Literal["supervisor"]]:
    result = economist_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="economist_agent")
            ]
        },
        goto="supervisor",
    )

evaluator_agent = create_react_agent(llm, tools=[textbook_reader],
                                 state_modifier="""
First, review the message you receive from economist_agent_node and identify if it mentions the following categories, healthcare, investments, technology, finance, construction, and real estate.
Next you will need to pass the tool a list for the path_list variable if the message from the economist_agent is in one of the categories.
The paths for each category is below:
Healthcare = CategoryTextbooks/The-Economics-of-Health-and-Health-Care.pdf
Real Estate = CategoryTextbooks/Book_ECON.pdf
Construction = CategoryTextbooks/ConstructionTextbook.pdf
Finance = CategoryTextbooks/PrinciplesofFinance-WEB.pdf
Technology = CategoryTextbooks/TechEconTextbook.pdf
Investments = CategoryTextbooks/InvestmentTextbook.pdf

Provide a comprehensive and be descriptive on what message you receive and what is found in the textbooks.  
Also provide as much information as possible that you find and if there are multiple categories in the question, have the category as the title then provide your explaination.
Once the textbooks have been reviewed you will do the following below:
- Find key phrases in the question and relate it to the textbook then provide context from the textbook and solutions to economic issues. Examples Provided below:
        - Asymmetrical information could be solved by intermediaries or rating agencies such as Moody's and Standard & Poor's informing market participants about securities risk. 
        - If businesses hire too few low-skilled workers after a minimum wage increase, the government can create exceptions for less-skilled workers. Governments can also impose taxes and subsidies as possible solutions. Subsidies can help encourage behavior that can result in positive externalities. Meanwhile, taxation can help cut down negative behavior. For example, placing a tax on tobacco can increase the cost of consumption, therefore making it more expensive for people to smoke.
- I want atleast 2 or 3 sentences from the textbook related to the keyword. 
- Then, to the best of your ability answer the question if you know anything else.
- Make sure to distinguish what you got from the textbook or what you get from the message provided
                                                    """)


def evaluator_node(state: State) -> Command[Literal["supervisor"]]:
    result = evaluator_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="evaluator_agent")
            ]
        },
        goto="supervisor",
    )

builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("economist_agent", economist_agent_node)
builder.add_node("evaluator_agent", evaluator_node)
graph = builder.compile()



# for s in graph.stream(
#     {"messages": [("user", f"""What is the current state in the construction and real estate industry?
#                 """)]}, subgraphs=True
#     ):
#     if list(s[1])[0] in ["economist_agent", "evaluator_agent"]:
#         print(list(s[1].values())[0]['messages'][0].content)
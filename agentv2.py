import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from resource import prompt



def agentWithTravilySearchTool():

    llm=ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # llm =ChatOpenAI(
    #     model="gpt-4o",
    #     temperature=0
    # )

    tool_list=[TavilySearch()]

    agent=create_agent(
        model=llm,
        tools=tool_list
    )

    agent_response = agent.invoke(
        {
            "messages": 
            HumanMessage(content=prompt)
        }
    )

    return agent_response
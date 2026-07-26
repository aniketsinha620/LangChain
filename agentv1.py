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
from responseStructure import ResponseStructure

tavily = TavilyClient()

@tool
def tool_function(city: str) -> {}:
    """
    Search LinkedIn and Naukri.com for current job opportunities.

    Use this tool whenever a user requests jobs, internships,
    hiring information, or career opportunities.

    The input should be the user's complete request. Do not remove
    important details such as skills, experience, location,
    technologies, or number of jobs requested.

    Return only the most relevant and recent job postings.
    Include:
    - Job title
    - Company
    - Location
    - Experience required
    - Required skills
    - Job URL
    - Short description
    """
    print("tool is called")
    result = tavily.search(
        query=f"Current weather in {city}"
    )

    return "\n".join(
        item["content"]
        for item in result.get("results", [])
    )

def agentWithCustomTool():

    llm=ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

    # llm =ChatOpenAI(
    #     model="gpt-4o",
    #     temperature=0
    # )


    tool_list=[tool_function]

    agent=create_agent(
        model=llm,
        tools=tool_list,
        response_format=ResponseStructure
    )

    agent_response = agent.invoke(
        {
            "messages": 
            HumanMessage(content=prompt)
        }
    )

    return agent_response
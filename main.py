import os
from dotenv import load_dotenv
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tavily import TavilyClient
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_tavily import TavilySearch
from agentv2 import agentWithTravilySearchTool
from agentv1 import agentWithCustomTool

load_dotenv()



    
def main():
    
    agent_response = agentWithTravilySearchTool()
    # agent_response = agentWithCustomTool()
    print(agent_response["structured_response"].model_dump_json(indent=4))


if __name__ == "__main__":
    main()

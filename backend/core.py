import os
from typing import Any, Dict


from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# chat_model = init_chat_model(
#     model="llama-3.3-70b-versatile",
#     model_provider="groq"
# )

chat_model = init_chat_model(
    model="gpt-4.1-mini",
    model_provider="openai"
)

embedding_model=OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vector_store=PineconeVectorStore(
    embedding=embedding_model,
    index_name=os.getenv("INDEX_NAME")
)


@tool(response_format="content_and_artifact")
def reterive_information(query:str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""


    print("*************reterive_information function is called*************")
    reterive=vector_store.as_retriever(search_kwargs={"k": 4})
    reterive_data=reterive.invoke(query)
    # print(reterive_data)

    serialized="".join(
        (
            f"Source:{docs.metadata.get('source','umknown')}\n\nContent:{docs.page_content}"
            for docs in reterive_data
        )
    )

    return serialized,reterive_data


def init_llm_chat(query:str)->Dict[str,Any]:
    """
    Run the RAG pipeline to answer a query using retrieved documentation.
    
    Args:
        query: The user's question
        
    Returns:
        Dictionary containing:
            - answer: The generated answer
            - context: List of retrieved documents
    """
    # Create the agent with retrieval tool
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    agent=create_agent(
        model=chat_model,
        tools=[reterive_information],
        system_prompt=system_prompt
    )

    inputs = {"messages": [{"role": "user", "content": query}]}

    response=agent.invoke(inputs)

    answer = response["messages"][-1].content
    
    # Extract context documents from ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)
    
    return {
        "answer": answer,
        "context": context_docs
    }



if __name__=="__main__":
    print("**********starting the program************")
    data=init_llm_chat("Guido van Rossum")
    print("data",data)


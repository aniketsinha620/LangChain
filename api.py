from resource import informations
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate 


def main_api(llm: str="groq"):

    print(f"{llm} chain is called")
    prompt_template="""
    You are a helpful assistant that summarizes information about a person {information}, give me these format:
    Summary:
    Three key points:
    """

    summary_prompt=PromptTemplate(
        input_variables=["information"],
        template=prompt_template
    )

    if llm=="groq":
        llm= ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    elif llm=="ollama":
        llm= ChatOllama(
            model="gemma3:270m",
            temperature=0
        )


    chain= summary_prompt | llm
    res=chain.invoke(
        input={"information": informations}
    )

    return res.content

import os
from operator import itemgetter
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


prompt=ChatPromptTemplate.from_template(
    """
     Answer the question based only on the following context:
    {context}

     Question: {question}

    Provide a detailed answer:
    """)

# prompt=ChatPromptTemplate.from_template(
#     """
#      You are the chat bot give the details answer to the below question

#      Question: {question}

#     Provide a detailed answer:
#     """)


def format_data(data:list):
    return "".join(ele.page_content for ele in data)

def main():
    print("Hello from langchain!")

    llm=ChatOpenAI()

    embedding_model=OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )


    vector_store=PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embedding_model,
    )

    reteriver=vector_store.as_retriever(
       search_kwargs={"k":3}
    )


    chain=(RunnablePassthrough.assign(
        context=itemgetter('question')|reteriver|format_data
    )|prompt|llm|StrOutputParser())

    response=chain.invoke({"question":"what aniket is currently doing"})

    # chain=prompt|llm|StrOutputParser()
    # response=chain.invoke({"question":"what aniket is currently doing"})

    return response




if __name__ == "__main__":
    data=main()
    print(data)

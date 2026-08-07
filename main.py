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

llm=ChatOpenAI()

embedding_model=OpenAIEmbeddings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

vector_store=PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embedding_model
    )

retriever=vector_store.as_retriever(
        search_kwargs={"k":3}
    )



def format_data(data:list):
    return "".join(ele.page_content for ele in data)



def rag_pipeline_without_lcel(query):

    print("="*60)
    docs=retriever.invoke(query)

    prompt_message=prompt.format_messages(
        context=format_data(docs),
        question=query
    )

    response=llm.invoke(prompt_message)
    return response.content


def rag_pipeline_with_lcel():
    print("="*60)

    chain= (RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_data
    )|prompt|llm|StrOutputParser())


    response=chain.invoke({"question":"what is pinecone vector store?"})

    print("data",response)

 





if __name__ == "__main__":
    # data=rag_pipeline_without_lcel("what is pinecone vector store?")
    data=rag_pipeline_with_lcel()
    print(data)
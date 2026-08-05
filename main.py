import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.messages import SystemMessage,HumanMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def main():

    llm=ChatOpenAI()

    embedding_model=OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vector_store=PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embedding_model
    )

    search_vector_store=vector_store.as_retriever(
        search_kwargs={"k": 3}
    )






if __name__ == "__main__":
    main()
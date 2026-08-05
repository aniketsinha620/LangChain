import os
from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore


load_dotenv()


def injection():
    print("Hello from langchain!")
    loader=UnstructuredLoader(
        file_path="C:\\project\\LLM\\LangChain\\mediumblog1.txt",
        max_characters=100000,
        chunking_strategy="basic",
    )

    data=loader.load()
    # print(f"Loaded {len(data)} documents.{data}")

    text_splitter=CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks=text_splitter.split_documents(data)

    # for i, chunk in enumerate(chunks, start=1):
    #     print(f"========== Chunk {i} ==========")
    #     print(chunk.page_content)
    #     print("\n")


    embedding_model=OpenAIEmbeddings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    print("ingestion....")

    vector_store=PineconeVectorStore.from_documents(
        chunks,
        embedding=embedding_model,
        index_name=os.getenv("INDEX_NAME"),
    )

    print("finish")



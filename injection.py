import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import (Colors, log_error, log_header, log_info, log_success,log_warning)

load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()



embedding_model=OpenAIEmbeddings(
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=4
)


vector_database=PineconeVectorStore(
    embedding=embedding_model,
    index_name=os.getenv("INDEX_NAME")
)

tavily_store=TavilyCrawl()

async def add_batch(batch: List[Document], batch_num: int,batches):
        try:
            await vector_database.aadd_documents(batch)
            log_success(
                f"VectorStore Indexing: Successfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
            )
        except Exception as e:
            log_error(f"VectorStore Indexing: Failed to add batch {batch_num} - {e}")
            return False
        return True

async def data_reteriver():

    log_header("DOCUMENT INJECTION FUNCTION CALLED")

    log_info(
        "🗺️ TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )

    # res = tavily_store.invoke(
    #     {
    #         "url": "https://docs.langchain.com/oss/python/langchain/overview/",
    #         "max_depth": 1,
    #         "extract_depth": "advanced",
    #     }
    # )    
    res = tavily_store.invoke(
        {
            "url": "https://en.wikipedia.org/wiki/LLM-as-a-Judge/",
            "extract_depth": "advanced",
        }
    )

    print("TYPE:", type(res))
    # print("RES:", res)

    if not res or not isinstance(res, dict):
        log_warning(f"Tavily crawl returned no structured results: {res}")
        return None

    if not res.get("results"):
        log_warning("Tavily crawl returned empty results")
        return None

    all_docs = []

    for item in res["results"]:
        all_docs.append(
            Document(
                page_content=item["raw_content"],
                metadata={"source": item["url"]},
            )
        )

    return all_docs

async def data_splitter(data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0,
    )

    chunks_data=text_splitter.split_documents(data)

    return chunks_data

async def index_documents_async(
    documents: List[Document],
    batch_size: int = 50
):
    log_header("VECTOR STORAGE PHASE")

    batches = [
        documents[i:i + batch_size]
        for i in range(0, len(documents), batch_size)
    ]

    log_info(
        f"📦 VectorStore Indexing: Split into {len(batches)} batches"
    )

    successful = 0

    for i, batch in enumerate(batches, start=1):
        result = await add_batch(
            batch,
            i,
            batches
        )

        if result:
            successful += 1

    if successful == len(batches):
        log_success(
            f"All batches processed successfully! "
            f"({successful}/{len(batches)})"
        )
    else:
        log_warning(
            f"Processed {successful}/{len(batches)} batches successfully"
        )

async def main():
    content_data=await data_reteriver()

    if not content_data:
        print("content data is not found...............")
        return

    chunks=await data_splitter(content_data)
    await index_documents_async(chunks)

if __name__=="__main__":
    print("..........injection started..........")
    asyncio.run(main())



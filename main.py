# from langchain_openai import ChatOpenAI
import os
from langchain import hub
from langchain_together import ChatTogether
#from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownTextSplitter
import pymupdf4llm
from pinecone import Pinecone, ServerlessSpec
from typing import cast
from dotenv import load_dotenv

import chainlit as cl

load_dotenv()

together_api_key = os.environ.get('TOGETHERAI_KEY')
pinecone_api_key = os.environ.get('PINECONE_API_KEY')
index_name = 'l10'
source_doc = 'data/123hk.pdf'
embeddings = HuggingFaceEmbeddings(model_name="infgrad/stella-base-zh-v2")


def rebuild_index(index_name: str) -> PineconeVectorStore:
    pc = Pinecone(api_key=pinecone_api_key)
    pc.delete_index(index_name)
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        ),
        deletion_protection="disabled"
    )
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)
    md_text = pymupdf4llm.to_markdown(source_doc)
    splitter = MarkdownTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = splitter.create_documents([md_text])
    vector_store.add_documents(documents=docs)
    return (vector_store)

# set the LANGCHAIN_API_KEY environment variable (create key in settings)


vector_store = rebuild_index(index_name)
retriever = vector_store.as_retriever()


@cl.on_chat_start
async def on_chat_start():
    # model = ChatOpenAI(streaming=True)
    model = ChatTogether(
        model="mistralai/Mixtral-8x22B-Instruct-v0.1",
        api_key=together_api_key
    )

    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
        print(docs)
        return "\n\n".join(doc.page_content for doc in docs)

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    # msg.send(runnable.invoke({"question": message.content}))
    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()

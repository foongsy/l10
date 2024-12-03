# from langchain_openai import ChatOpenAI
import os
from langchain_together import ChatTogether
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
from typing import cast
from dotenv import load_dotenv

import chainlit as cl

load_dotenv()

together_api_key = os.environ['TOGETHERAI_KEY']

@cl.on_chat_start
async def on_chat_start():
    # model = ChatOpenAI(streaming=True)
    model = ChatTogether(
        model="meta-llama/Llama-3-70b-chat-hf",
        api_key=together_api_key
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You're a very knowledgeable historian who provides accurate and eloquent answers to historical questions.",
            ),
            ("human", "{question}"),
        ]
    )
    runnable = prompt | model | StrOutputParser()
    cl.user_session.set("runnable", runnable)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cast(Runnable, cl.user_session.get("runnable"))  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()
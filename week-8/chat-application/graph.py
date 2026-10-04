import uuid
from datetime import datetime

import aiosqlite
import asyncio

from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.sqlite.aio import AsyncSqliteStore

from langchain.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

DB_PATH = "test.db"


# -------------------------
# LLM
# -------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=500,
    temperature=0.1,
)
    
# -------------------------
# State
# -------------------------

class State(MessagesState):
    pass


# -------------------------
# Chat Node
# -------------------------

async def chat_node(state: State, config):

    thread_id = config["configurable"]["thread_id"]

    print("Thread:", thread_id)

    response = await llm.ainvoke(
        state["messages"]
    )

    return {
        "messages": [response]
    }


# -------------------------
# Graph
# -------------------------

graph = StateGraph(State)

graph.add_node("chat_node",chat_node)
graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

# -------------------------
# Get Messages
# -------------------------

async def get_thread_messages(
    thread_id: str
):

    async with AsyncSqliteSaver.from_conn_string(
        DB_PATH
    ) as checkpointer:

        workflow = graph.compile(
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        state = await workflow.aget_state(
            config
        )

        return state.values.get(
            "messages",
            []
        )

# -------------------------
# Invoke Chat
# -------------------------

async def invoke_workflow(
    message: str,
    thread_id: str
):

    async with AsyncSqliteSaver.from_conn_string(
        DB_PATH
    ) as checkpointer:

        workflow = graph.compile(
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        response = await workflow.invoke(
            {
                "messages": [HumanMessage(content=message)]
            },
            config=config,
        )

    # Update thread timestamp
    async with aiosqlite.connect(DB_PATH) as db:

        await db.execute(
            """
            UPDATE threads
            SET updated_at = ?
            WHERE thread_id = ?
            """,
            (
                datetime.now().isoformat(),
                thread_id,
            )
        )

        await db.commit()

    return response

# -------------------------
# Stream Chat
# -------------------------
async def stream_workflow(message, thread_id):

    async with AsyncSqliteSaver.from_conn_string(DB_PATH) as checkpointer:

        workflow = graph.compile(
            checkpointer=checkpointer
        )

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        async for chunk in workflow.astream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": message,
                    }
                ]
            },
            config=config,
            stream_mode="messages",
        ):
            message_chunk, metadata = chunk

            if message_chunk.content:
                yield message_chunk.content

    # Update thread timestamp
    async with aiosqlite.connect(DB_PATH) as db:

        await db.execute(
            """
            UPDATE threads
            SET updated_at = ?
            WHERE thread_id = ?
            """,
            (
                datetime.now().isoformat(),
                thread_id,
            ),
        )

        await db.commit()
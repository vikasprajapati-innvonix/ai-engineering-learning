import sqlite3
import uuid
from contextlib import contextmanager
from typing import NotRequired, TypedDict

from config import MEMORY_DB
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.store.sqlite import SqliteStore
from pydantic import BaseModel, Field

load_dotenv()

MEMORY_PROMPT = """You are a memory extraction assistant.

Extract facts from the user's message that are worth remembering across
future conversations (identity, role, stack, goals, constraints, stable preferences).

Rules:
- Only extract what the user explicitly stated. Never infer or invent.
- Skip anything already covered by the existing memories below.
- Skip one-off, transient details.
- If nothing qualifies, return an empty list.

Existing memories:
{existing}

User message:
{query}
"""

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=2000,
    temperature=0.1,
)


class Memory(BaseModel):
    statement: str = Field(..., description="A concise statement of the memory.")
    reason: str = Field(..., description="Why it is useful for future conversations.")


class MemoryExtraction(BaseModel):
    """LLM output schema: zero or more memories."""

    memories: list[Memory] = Field(default_factory=list)


class MemoryState(TypedDict):
    query: str                                      # required: the input message
    relevant_memories: NotRequired[list[Memory]]    # set by retrieve_memories node
    new_memories: NotRequired[list[Memory]]         # set by extract_memories node


structured_llm = llm.with_structured_output(MemoryExtraction)


def _namespace(config: RunnableConfig) -> tuple[str, str]:
    return ("user", config["configurable"]["user_id"])


def retrieve_memories(state: MemoryState, config: RunnableConfig, store):
    items = store.search(_namespace(config), query=state["query"], limit=5)
    print(f"items: {items}")
    return {
        "relevant_memories": [
            Memory(
                statement=i.value["data"],
                reason=i.value.get("reason", ""),
            )
            for i in items
        ]
    }


def extract_memories(state: MemoryState, config: RunnableConfig, store):
    existing = state.get("relevant_memories", [])
    print(f"existing: {existing}")
    prompt = MEMORY_PROMPT.format(
        query=state["query"],
        existing="\n".join(f"- {m.statement}" for m in existing) or "(none)",
    )

    result = structured_llm.invoke(prompt)
    print(f"result: {result}")
    seen = {m.statement.strip().lower() for m in existing}
    namespace = _namespace(config)
    saved: list[Memory] = []

    for m in result.memories:
        key = m.statement.strip().lower()
        if key in seen:
            continue
        seen.add(key)
        store.put(
            namespace,
            str(uuid.uuid4()),
            {"data": m.statement, "reason": m.reason},
        )
        saved.append(m)
    print(f"saved: {saved}")
    return {"new_memories": saved}


builder = StateGraph(MemoryState)
builder.add_node("retrieve_memories", retrieve_memories)
builder.add_node("extract_memories", extract_memories)
builder.add_edge(START, "retrieve_memories")
builder.add_edge("retrieve_memories", "extract_memories")
builder.add_edge("extract_memories", END)


@contextmanager
def open_store():
    embedding = MistralAIEmbeddings(model="mistral-embed")
    index_config = {"embed": embedding, "dims": 1024, "fields": ["data"]}
    # check_same_thread=False: LangGraph may access from internal worker threads
    conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False, isolation_level=None)
    store = SqliteStore(conn, index=index_config)
    store.setup()
    try:
        yield store
    finally:
        conn.close()

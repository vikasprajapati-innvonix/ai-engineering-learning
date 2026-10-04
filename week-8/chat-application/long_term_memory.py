import aiosqlite
import asyncio

from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph

from pydantic import BaseModel,Field
from typing import Annotated, TypedDict

from langchain_core.prompts import PromptTemplate

import operator

from langgraph.store.sqlite.aio import AsyncSqliteStore

MEMORY_PROMPT = """
You are a memory extraction assistant.

Your task is to analyze the user's message and conversation context
and identify information that is useful to remember across future
conversations.

Return ONLY the memories that should be stored.

For each memory, return:
1. A concise statement of the memory.
2. Why it is useful for future conversations.

If there is no information worth remembering, return an empty list.

Do not invent or infer memories that the user did not explicitly provide.

Query: {query}
---
Previous memories: {previous_memories}

Rules: 
    - Do not invent or infer memories that the user did not explicitly provide.
    - Return ONLY the memories that should be stored.
    - Do not provide explanations, roadmaps, lists, or detailed steps unless explicitly asked.
    - Do not repeat previous conversation unnecessarily.
"""

DB_PATH = "test.db"

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=500,
    temperature=0.1,
)

class Memory(BaseModel):
    statement: str = Field(...,description="A concise statement of the memory.")
    reason: str = Field(...,description="Why it is useful for future conversations.")


class MemoryValidation(BaseModel):
    memories: Annotated[list[Memory],operator.add] = []

class Query(TypedDict):
    query: str

structured_llm = llm.with_structured_output(MemoryValidation)

def extract_memories(state: Query,config,store):
    query = state['query']

    previous_memories = store.search(
        ('user',config['configurable']['thread_id']),
        query=query,
    )

    response = structured_llm.invoke(
        PromptTemplate.from_template(MEMORY_PROMPT).format(
            query=query,
            previous_memories="\n".join(f"- {item.value.get('data','')}" for item in previous_memories),
        )
    )

    print(response)

    return {
        "memories": response.memories
    }



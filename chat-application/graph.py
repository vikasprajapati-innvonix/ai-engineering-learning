import sqlite3
from contextlib import contextmanager

from config import CHAT_DB, USER_ID
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from long_term_memory import Memory, open_store
from long_term_memory import builder as memory_builder
from mcp_tool_bind import get_all_available_tools
from tool_logger import ToolLogger

load_dotenv()


# -------------------------
# LLM (chat only; the extractor has its own instance)
# -------------------------
chat_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    max_tokens=2000,
    temperature=0.3,
)

RECURSION_LIMIT = 15  # caps the chat <-> tools loop


# -------------------------
# State
# -------------------------
class State(MessagesState):
    memory_context: str  # overwritten every turn, never accumulates


# -------------------------
# Nodes
# -------------------------
BASE_SYSTEM_PROMPT = "You are a helpful assistant."
LOAD_MEMORY_NODE = "load_memory"
CHAT_NODE = "chat_node"
TOOLS_NODE = "tools"


def format_memories(memories: list[Memory]) -> str:
    if not memories:
        return ""
    lines = "\n".join(f"- {m.statement}" for m in memories)
    return f"Known facts about the user from past conversations:\n{lines}"


def last_human_text(messages) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""


def load_memory(state: State, config: RunnableConfig, store):
    """Runs once per user turn, before the chat/tools loop."""
    user_id = config["configurable"]["user_id"]
    result = memory_builder.compile(store=store).invoke(
        {"query": last_human_text(state["messages"])},
        config={"configurable": {"user_id": user_id}},
    )
    return {"memory_context": format_memories(result.get("relevant_memories", []))}


# -------------------------
# Graph
# -------------------------
def build_graph(tools: list) -> StateGraph:
    llm_with_tools = chat_llm.bind_tools(tools) if tools else chat_llm
    print("tools", tools)

    def chat_node(state: State):

        system = BASE_SYSTEM_PROMPT
        if state.get("memory_context"):
            system += "\n\n" + state["memory_context"]

        # Local list only: the SystemMessage must not be returned into state
        messages = [SystemMessage(content=system), *state["messages"]]

        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e:
            # Groq sometimes rejects malformed tool calls from gpt-oss
            if "tool_use_failed" not in str(e):
                raise
            response = chat_llm.invoke(messages)

        return {"messages": [response]}

    g = StateGraph(State)
    g.add_node(LOAD_MEMORY_NODE, load_memory)
    g.add_node(CHAT_NODE, chat_node)
    # handle_tool_errors=True: if a tool raises (DNS error, timeout, etc.)
    # LangGraph catches it and returns a ToolMessage with the error text,
    # so the LLM can respond gracefully instead of the graph crashing.
    g.add_node(TOOLS_NODE, ToolNode(tools, handle_tool_errors=True))

    g.add_edge(START, LOAD_MEMORY_NODE)
    g.add_edge(LOAD_MEMORY_NODE, CHAT_NODE)
    g.add_conditional_edges(CHAT_NODE, tools_condition)  # -> "tools" or END
    g.add_edge(TOOLS_NODE, CHAT_NODE)

    return g


# -------------------------
# Compile Graph
# -------------------------
@contextmanager
def compile_graph(store=None, tools=None):
    # isolation_level=None = autocommit: disables Python's implicit BEGIN so
    # SqliteSaver can manage its own transactions without conflict.
    conn = sqlite3.connect(str(CHAT_DB), check_same_thread=False, isolation_level=None)
    try:
        checkpointer = SqliteSaver(conn)
        checkpointer.setup()
        workflow = build_graph(tools or []).compile(
            checkpointer=checkpointer, store=store
        )
        print(workflow.get_graph().draw_ascii())
        yield workflow
    finally:
        conn.close()


def load_tools() -> list:
    try:
        tools = get_all_available_tools()
        print(f"Loaded {len(tools)} tools: {[t.name for t in tools]}")
        return tools

    except Exception as e:
        print(f"MCP TOOL LOADING FAILED: {e!r}")
        raise  # Don't hide the actual failure while debugging


# -------------------------
# Get Messages (reads checkpoints only, no store or tools needed)
# -------------------------
def get_thread_messages(thread_id: str):
    with compile_graph() as workflow:
        config = {"configurable": {"thread_id": thread_id}}
        state = workflow.get_state(config)
        return state.values.get("messages", [])


# -------------------------
# Stream Chat
# -------------------------
def stream_workflow(message: str, thread_id: str):
    """Sync generator — compatible directly with st.write_stream()."""
    tools = load_tools()
    with open_store() as store, compile_graph(store, tools) as workflow:
        config = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {
                "thread_id": thread_id,
                "user_id": USER_ID,
            },
            "callbacks": [ToolLogger()],
        }
        for mode, chunk in workflow.stream(
            {"messages": [HumanMessage(content=message)]},
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk

                if metadata.get("langgraph_node") != CHAT_NODE:
                    continue

                if message_chunk.content:
                    yield {
                        "type": "token",
                        "text": message_chunk.content,
                    }

            elif mode == "updates":
                for node_name, node_update in chunk.items():
                    # Show which graph node is running
                    yield {
                        "type": "status",
                        "text": f"Running: {node_name}",
                    }

                    for msg in node_update.get("messages", []):
                        reasoning = msg.additional_kwargs.get("reasoning_content")
                        if reasoning:
                            yield {
                                "type": "reasoning",
                                "text": reasoning,
                            }

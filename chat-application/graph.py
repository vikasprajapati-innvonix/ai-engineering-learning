import sqlite3
from contextlib import contextmanager

from config import CHAT_DB, USER_ID
from dotenv import load_dotenv
from hitl import approval_node, route_after_approval, route_after_chat
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
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
    approval: str


# -------------------------
# Nodes
# -------------------------
BASE_SYSTEM_PROMPT = "You are a helpful assistant."
LOAD_MEMORY_NODE = "load_memory"
APPROVAL_NODE = "approval"
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
    g.add_node(APPROVAL_NODE, approval_node)
    g.add_node(CHAT_NODE, chat_node)
    # handle_tool_errors=True: if a tool raises (DNS error, timeout, etc.)
    # LangGraph catches it and returns a ToolMessage with the error text,
    # so the LLM can respond gracefully instead of the graph crashing.
    g.add_node(TOOLS_NODE, ToolNode(tools, handle_tool_errors=True))

    g.add_edge(START, LOAD_MEMORY_NODE)
    g.add_edge(LOAD_MEMORY_NODE, CHAT_NODE)
    g.add_conditional_edges(
        CHAT_NODE,
        route_after_chat,
        {"__end__": END, "tools": TOOLS_NODE, "approval": APPROVAL_NODE},
    )
    # After approval_node: approved → run the tool, rejected → back to chat
    g.add_conditional_edges(
        APPROVAL_NODE,
        route_after_approval,
        {"tools": TOOLS_NODE, "chat_node": CHAT_NODE},
    )
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
        
        # Store graph image in file
        # with open("./graph.png", "wb") as f:
        #     f.write(workflow.get_graph().draw_mermaid_png())
        
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


def _interrupt_text(value) -> str:
    if isinstance(value, dict):
        return value.get("question") or str(value)
    return str(value)

def get_pending_approval(thread_id: str) -> str | None:
    """
    Read graph state from the checkpointer only — no need to load tools or
    open the memory store just to check if an interrupt is pending.
    """
    with compile_graph() as workflow:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = workflow.get_state(config)
        for task in snapshot.tasks:
            for intr in task.interrupts:
                return _interrupt_text(intr.value)
    return None

# -------------------------
# Stream Chat
# -------------------------
def stream_workflow(message: str | None, thread_id: str, decision=None):
    if decision is not None:
        graph_input = Command(resume=decision)
    elif message:
        graph_input = {"messages": [HumanMessage(content=message)]}
    else:
        return

    tools = load_tools()

    with open_store() as store, compile_graph(store, tools) as workflow:
        config: RunnableConfig = {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {
                "thread_id": thread_id,
                "user_id": USER_ID,
            },
            "callbacks": [ToolLogger()],
        }

        for mode, chunk in workflow.stream(
            graph_input,
            config=config,
            stream_mode=["updates", "messages"],
        ):
            if mode == "messages":
                message_chunk, metadata = chunk
                if (
                    metadata.get("langgraph_node") == CHAT_NODE
                    and message_chunk.content
                ):
                    yield {"type": "token", "text": message_chunk.content}

            elif mode == "updates":
                for node_name, node_update in chunk.items():
                    if node_name == "__interrupt__":
                        for intr in node_update:
                            yield {
                                "type": "approval",
                                "text": _interrupt_text(intr.value),
                            }
                        continue

                    yield {"type": "status", "text": f"Running: {node_name}"}

                    if not node_update:
                        continue

                    for msg in node_update.get("messages", []):
                        reasoning = msg.additional_kwargs.get("reasoning_content")
                        if reasoning:
                            yield {"type": "reasoning", "text": reasoning}

        # Safety-net: also check state after the stream ends in case the
        # __interrupt__ update was not emitted mid-stream.
        snapshot = workflow.get_state(config)
        for task in snapshot.tasks:
            for intr in task.interrupts:
                yield {"type": "approval", "text": _interrupt_text(intr.value)}

import asyncio
import concurrent.futures
import os

from dotenv import load_dotenv
from fastmcp import Client, ClientGroup
from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter
from langchain_core.tools import StructuredTool
from tools import tools_list

load_dotenv()

PROJECT_ABSOLUTE_PATH = os.environ["PROJECT_ABSOLUTE_PATH"]


def create_mcp_clients():
    task_server_transport = StdioTransport(
        command=(f"{PROJECT_ABSOLUTE_PATH}/ai-engineering-learning/.venv/bin/python"),
        args=[
            f"{PROJECT_ABSOLUTE_PATH}/ai-engineering-learning/chat-application/MCP/task_server.py"
        ],
    )
    task_manager_client = Client(transport=task_server_transport)

    exchange_server_transport = StdioTransport(
        command=(f"{PROJECT_ABSOLUTE_PATH}/ai-engineering-learning/.venv/bin/python"),
        args=[
            f"{PROJECT_ABSOLUTE_PATH}/ai-engineering-learning/chat-application/MCP/exchange_rate_conversion_server.py"
        ],
    )
    exchange_rate_converter_client = Client(transport=exchange_server_transport)

    return ClientGroup(
        {
            "task_manager": task_manager_client,
            "exchange_rate_converter": exchange_rate_converter_client,
        }
    )


# -------------------------
# Schema discovery (once at load time)
# -------------------------


async def _async_get_tool_schemas():
    """
    Open a fresh MCP session, read tool metadata (name/description/args_schema),
    then close the session immediately.

    We only keep the *schemas* — NOT the tool coroutines themselves.
    Coroutines returned by list_tools() hold references to the adapter's
    stdio subprocess, which is killed when the `async with` block exits.
    Calling those coroutines later would hit 'Event loop is closed'.
    """
    clients = create_mcp_clients()
    async with MCPAdapter(clients) as adapter:
        tools = await adapter.list_tools()
        return [(t.name, t.description, t.args_schema) for t in tools]


# -------------------------
# Per-call execution (fresh session every time)
# -------------------------


async def _async_call_mcp_tool(tool_name: str, kwargs: dict):
    """
    Open a brand-new MCP session, call the named tool, close the session.

    A fresh session is required because the stdio subprocess is process-bound
    to a single event loop. Each call gets its own subprocess and loop.
    """
    clients = create_mcp_clients()
    async with MCPAdapter(clients) as adapter:
        tools = await adapter.list_tools()
        for tool in tools:
            if tool.name == tool_name:
                return await tool.coroutine(**kwargs)
    raise ValueError(f"MCP tool '{tool_name}' not found in server")


def _run_in_thread(coro):
    """Run an async coroutine in a background thread with its own event loop."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


# -------------------------
# Build callable StructuredTools from schemas
# -------------------------


def _make_mcp_tool(name: str, description: str, args_schema) -> StructuredTool:
    """
    Create a StructuredTool that opens a fresh MCP session on every invocation.

    Both sync (func) and async (coroutine) paths are provided:
    - func      → used by LangGraph's sync ToolNode
    - coroutine → used if the graph is ever switched back to async
    """

    def sync_run(**kwargs):
        return _run_in_thread(_async_call_mcp_tool(name, kwargs))

    async def async_run(**kwargs):
        return await _async_call_mcp_tool(name, kwargs)

    return StructuredTool(
        name=name,
        description=description,
        args_schema=args_schema,
        func=sync_run,
        coroutine=async_run,
    )


# -------------------------
# Public API
# -------------------------


def get_all_available_tools() -> list:
    """
    Sync public API.

    1. Discovers MCP tool schemas (names, descriptions, parameter types) by
       opening one short-lived session in a background thread.
    2. Builds StructuredTools whose sync/async run methods each open their own
       fresh MCP session — no stale event loop or dead subprocess references.
    """
    schemas = _run_in_thread(_async_get_tool_schemas())
    mcp_tools = [_make_mcp_tool(name, desc, schema) for name, desc, schema in schemas]
    return tools_list + mcp_tools

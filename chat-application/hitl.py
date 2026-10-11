from langchain_core.messages import ToolMessage
from langgraph.prebuilt import tools_condition
from langgraph.types import interrupt

# Tools that require user approval before execution
APPROVAL_REQUIRED_TOOLS = ["task_manager_add_task", "task_manager_delete_task"]


# -------------------------
# Routing: after chat_node
# -------------------------

def route_after_chat(state) -> str:
    """
    Extends LangGraph's default tools_condition:
    - If no tool call → END
    - If tool call needs approval → "approval"
    - Otherwise → "tools" (execute immediately)
    """
    route = tools_condition(state)

    if route == "__end__":
        return "__end__"

    last_message = state["messages"][-1]
    if any(call["name"] in APPROVAL_REQUIRED_TOOLS for call in last_message.tool_calls):
        return "approval"

    return "tools"


# -------------------------
# Routing: after approval_node
# -------------------------

def route_after_approval(state) -> str:
    """
    After the user decides:
    - "approved" → run the tools node to execute the call
    - "rejected" → back to chat_node so the LLM can inform the user
    """
    return "tools" if state.get("approval") == "approved" else "chat_node"


# -------------------------
# Approval node
# -------------------------

def approval_node(state):
    """
    Pauses the graph with interrupt() and waits for a human decision.

    On resume:
    - "approve" → sets approval="approved", tools node will execute the call
    - anything else → sets approval="rejected", injects a ToolMessage so the
      LLM understands the action was blocked (not a tool error)
    """
    last_message = state["messages"][-1]
    tool_call = last_message.tool_calls[-1]
    tool_name = tool_call["name"]
    args = tool_call.get("args", {})

    # Build a human-readable question depending on the tool
    if tool_name == "task_manager_add_task":
        question = (
            f"Add task \"{args.get('title', '?')}\"?\n"
            f"Description: {args.get('description', '')}"
        )
    elif tool_name == "task_manager_delete_task":
        question = f"Permanently delete task #{args.get('task_id', '?')}?"
    else:
        question = f"Approve action '{tool_name}' with args: {args}?"

    # Pause here — graph is suspended until Command(resume=decision) arrives
    decision = interrupt({"question": question, "tool_call": tool_call})

    if decision == "approve":
        return {"approval": "approved"}

    # Rejected: inject a ToolMessage so the LLM sees why it can't proceed.
    # Without this, the LLM has a dangling tool_call in its history with no
    # response — which causes a validation error on the next LLM call.
    rejection_msg = ToolMessage(
        content="The user rejected this action. Do not retry it automatically.",
        tool_call_id=tool_call["id"],
    )
    return {"approval": "rejected", "messages": [rejection_msg]}

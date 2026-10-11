import streamlit as st
from db_operation import (
    create_thread,
    get_threads,
    init_db,
)
from graph import (
    get_thread_messages,
    stream_workflow,
)

# =========================================================
# One-time DB init (idempotent: CREATE TABLE IF NOT EXISTS)
# =========================================================

init_db()


# =========================================================
# Session State
# =========================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("Chats")


# ---------------------------------------------------------
# New Chat
# ---------------------------------------------------------

if st.sidebar.button(
    "New Chat",
    icon=":material/add:",
    use_container_width=True,
):
    st.session_state.thread_id = None
    st.rerun()


st.sidebar.divider()
st.sidebar.write("Threads")


# ---------------------------------------------------------
# Get Threads
# ---------------------------------------------------------

threads = get_threads()


# ---------------------------------------------------------
# Display Threads
# ---------------------------------------------------------

for thread in threads:
    thread_id = thread["thread_id"]

    title = thread["title"] or "Untitled"

    is_active = thread_id == st.session_state.thread_id

    if st.sidebar.button(
        title,
        type="primary" if is_active else "secondary",
        key=f"thread_{thread_id}",
        use_container_width=True,
    ):
        st.session_state.thread_id = thread_id

        st.rerun()


# =========================================================
# Main Chat
# =========================================================

st.title("Chat Application")


# ---------------------------------------------------------
# Load Current Conversation
# ---------------------------------------------------------

if st.session_state.thread_id is not None:
    messages = get_thread_messages(st.session_state.thread_id)

    for message in messages:
        if message.type == "human":
            st.chat_message("user").write(message.content)

        elif message.type == "ai":
            st.chat_message("assistant").write(message.content)


# =========================================================
# Chat Input
# =========================================================

query = st.chat_input("Type your message here...")


if query:
    # -----------------------------------------------------
    # Create thread on first message
    # -----------------------------------------------------

    if st.session_state.thread_id is None:
        thread = create_thread(query)

        st.session_state.thread_id = thread["thread_id"]

    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    st.chat_message("user").write(query)

    # -----------------------------------------------------
    # Stream assistant
    # -----------------------------------------------------

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        answer_placeholder = st.empty()
        answer = ""

        for event in stream_workflow(
            message=query,
            thread_id=st.session_state.thread_id,
        ):
            if event["type"] == "status":
                status.write(event["text"])

            elif event["type"] == "reasoning":
                status.write(f"💭 {event['text']}")

            elif event["type"] == "token":
                answer += event["text"]
                answer_placeholder.markdown(answer)

        status.update(label="Done", state="complete", expanded=False)

    # -----------------------------------------------------
    # Reload
    # -----------------------------------------------------

    st.rerun()

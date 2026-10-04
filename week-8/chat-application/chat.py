import asyncio

import streamlit as st

from graph import (
    get_thread_messages,
    stream_workflow,
)

from db_operation import (
    create_thread,
    get_threads,
)

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
    "➕ New Chat",
    use_container_width=True,
):
    st.session_state.thread_id = None
    st.rerun()


st.sidebar.divider()


# ---------------------------------------------------------
# Get Threads
# ---------------------------------------------------------

threads = asyncio.run(
    get_threads()
)


# ---------------------------------------------------------
# Display Threads
# ---------------------------------------------------------

for thread in threads:

    thread_id = thread["thread_id"]

    title = thread["title"] or "Untitled"

    is_active = (
        thread_id == st.session_state.thread_id
    )

    # Add indicator to active thread
    label = (
        "👉 " if is_active else ""
    ) + title

    if st.sidebar.button(
        label,
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

    messages = asyncio.run(
        get_thread_messages(
            st.session_state.thread_id
        )
    )

    for message in messages:

        if message.type == "human":

            st.chat_message(
                "user"
            ).write(
                message.content
            )

        elif message.type == "ai":

            st.chat_message(
                "assistant"
            ).write(
                message.content
            )


# =========================================================
# Chat Input
# =========================================================

query = st.chat_input(
    "Type your message here..."
)


if query:

    # -----------------------------------------------------
    # Create thread on first message
    # -----------------------------------------------------

    if st.session_state.thread_id is None:

        thread = asyncio.run(
            create_thread(query)
        )

        st.session_state.thread_id = (
            thread["thread_id"]
        )


    # -----------------------------------------------------
    # Show user message
    # -----------------------------------------------------

    st.chat_message(
        "user"
    ).write(query)


    # -----------------------------------------------------
    # Stream assistant
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write_stream(
            stream_workflow(
                message=query,
                thread_id=st.session_state.thread_id,
            )
        )


    # -----------------------------------------------------
    # Reload
    # -----------------------------------------------------

    st.rerun()
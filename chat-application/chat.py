import streamlit as st
from db_operation import create_thread, get_threads, init_db
from graph import get_pending_approval, get_thread_messages, stream_workflow

init_db()

ss = st.session_state
ss.setdefault("thread_id", None)
ss.setdefault("pending_decision", None)  # "approve" | "reject" | None


def switch_thread(thread_id):
    ss.thread_id = thread_id
    ss.pending_decision = None  # never carry a decision across threads


# =========================================================
# Sidebar
# =========================================================

st.sidebar.title("Chats")

if st.sidebar.button("New Chat", icon=":material/add:", use_container_width=True):
    switch_thread(None)
    st.rerun()

st.sidebar.divider()
st.sidebar.write("Threads")

for thread in get_threads():
    tid = thread["thread_id"]
    if st.sidebar.button(
        thread["title"] or "Untitled",
        type="primary" if tid == ss.thread_id else "secondary",
        key=f"thread_{tid}",
        use_container_width=True,
    ):
        switch_thread(tid)
        st.rerun()

# =========================================================
# Main
# =========================================================

st.title("Chat Application")

# --- History (from the checkpointer) ---------------------
if ss.thread_id is not None:
    for m in get_thread_messages(ss.thread_id):
        if m.type == "human":
            st.chat_message("user").write(m.content)
        elif m.type == "ai" and m.content:  # skip tool-call-only turns
            st.chat_message("assistant").write(m.content)

# --- Approval UI (derived from graph state) --------------
pending_approval = None
if ss.thread_id is not None and ss.pending_decision is None:
    pending_approval = get_pending_approval(ss.thread_id)

if pending_approval:
    with st.chat_message("assistant"):
        st.warning(pending_approval)
        col1, col2 = st.columns(2)
        if col1.button("Approve", key=f"approve_{ss.thread_id}"):
            ss.pending_decision = "approve"
            st.rerun()
        if col2.button("Reject", key=f"reject_{ss.thread_id}"):
            ss.pending_decision = "reject"
            st.rerun()

# --- Single chat input -----------------------------------
query = st.chat_input(
    "Type your message here...",
    disabled=bool(pending_approval),
)

decision = ss.pending_decision

# =========================================================
# Run graph
# =========================================================

if query or decision:
    if ss.thread_id is None:
        ss.thread_id = create_thread(query)["thread_id"]

    if query:
        st.chat_message("user").write(query)

    ss.pending_decision = None  # consume before streaming

    needs_rerun = False

    with st.chat_message("assistant"):
        status = st.status("Thinking...", expanded=True)
        answer_placeholder = st.empty()
        answer = ""

        for event in stream_workflow(
            message=query,
            thread_id=ss.thread_id,
            decision=decision,
        ):
            if event["type"] == "status":
                status.write(event["text"])
            elif event["type"] == "reasoning":
                status.write(f"💭 {event['text']}")
            elif event["type"] == "token":
                answer += event["text"]
                answer_placeholder.markdown(answer)
            elif event["type"] == "approval":
                needs_rerun = True

        status.update(
            label="Approval required" if needs_rerun else "Done",
            state="complete",
            expanded=False,
        )

    # Rerun only so the approval buttons render; history reloads from DB.
    if needs_rerun:
        st.rerun()

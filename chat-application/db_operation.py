import sqlite3
from datetime import datetime
from uuid import uuid4

from config import CHAT_DB

# -------------------------
# Init DB
# -------------------------

def init_db():
    with sqlite3.connect(CHAT_DB) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        db.commit()


# -------------------------
# Create Thread
# -------------------------

def create_thread(title="New Chat"):

    thread_id = str(uuid4())

    now = datetime.now().isoformat()

    with sqlite3.connect(CHAT_DB) as db:
        db.execute(
            """
            INSERT INTO threads (
                thread_id,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                thread_id,
                title,
                now,
                now,
            ),
        )
        db.commit()

    return {
        "thread_id": thread_id,
        "title": title,
    }


# -------------------------
# Get Threads
# -------------------------

def get_threads():

    with sqlite3.connect(CHAT_DB) as db:
        db.row_factory = sqlite3.Row

        cursor = db.execute(
            """
            SELECT
                thread_id,
                title,
                created_at,
                updated_at
            FROM threads
            ORDER BY updated_at DESC
            """
        )

        rows = cursor.fetchall()

    return [dict(row) for row in rows]

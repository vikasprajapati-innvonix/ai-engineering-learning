import aiosqlite
import asyncio

DB_PATH = "test.db"

async def init_db():

    async with aiosqlite.connect(DB_PATH) as db:

        await db.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        await db.commit()

# -------------------------
# Create Thread
# -------------------------

async def create_thread(
    title="New Chat"
):

    thread_id = str(uuid.uuid4())

    now = datetime.now().isoformat()

    async with aiosqlite.connect(DB_PATH) as db:

        await db.execute(
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
            )
        )

        await db.commit()

    return {
        "thread_id": thread_id,
        "title": title,
    }


# -------------------------
# Get Threads
# -------------------------

async def get_threads():

    async with aiosqlite.connect(DB_PATH) as db:

        db.row_factory = aiosqlite.Row

        cursor = await db.execute(
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

        rows = await cursor.fetchall()

    return [
        dict(row)
        for row in rows
    ]

asyncio.run(init_db())

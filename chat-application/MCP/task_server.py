from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from fastmcp import FastMCP

mcp = FastMCP("Task Server")


DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

TASKS_DB = DATA_DIR / "tasks.db"

@asynccontextmanager
async def open_store():
    async with aiosqlite.connect(TASKS_DB) as db:
        # Ensure the table exists before any operation
        await db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL
            )
        """)
        await db.commit()
        yield db


@mcp.tool
async def get_tasks() -> list[dict]:
    """
    Retrieve all tasks stored in the database.

    Returns a list of task objects. Each object has three fields:
    - id (int): unique numeric identifier assigned automatically
    - title (str): short name of the task
    - description (str): full details of the task

    Use this before updating or deleting so you know the correct task id.
    """
    async with open_store() as db:
        cursor = await db.execute("SELECT id, title, description FROM tasks")
        rows = await cursor.fetchall()
        return [{"id": row[0], "title": row[1], "description": row[2]} for row in rows]


@mcp.tool
async def add_task(title: str, description: str) -> dict:
    """
    Create a new task and save it to the database.

    Parameters
    ----------
    title : str
        A short, human-readable name for the task. Keep it under 80 characters.
        Example: "Fix login bug"
    description : str
        Full details about the task — what needs to be done and why.
        Example: "The login form rejects valid emails containing a '+' character."

    Returns
    -------
    dict
        {"status": "created", "title": <title you provided>}
    """
    async with open_store() as db:
        await db.execute(
            "INSERT INTO tasks (title, description) VALUES (?, ?)",
            (title, description),
        )
        await db.commit()
    return {"status": "created", "title": title}


@mcp.tool
async def update_task(task_id: int, title: str, description: str) -> dict:
    """
    Update the title and description of an existing task.

    Parameters
    ----------
    task_id : int
        The numeric id of the task to update.
        Call get_tasks() first to find the correct id.
    title : str
        New short name for the task.
    description : str
        New full description for the task.

    Returns
    -------
    dict
        {"status": "updated", "id": <task_id>}
    """
    async with open_store() as db:
        await db.execute(
            "UPDATE tasks SET title = ?, description = ? WHERE id = ?",
            (title, description, task_id),
        )
        await db.commit()
    return {"status": "updated", "id": task_id}


@mcp.tool
async def delete_task(task_id: int) -> dict:
    """
    Permanently delete a task from the database.

    Parameters
    ----------
    task_id : int
        The numeric id of the task to delete.
        Call get_tasks() first to find the correct id.

    Returns
    -------
    dict
        {"status": "deleted", "id": <task_id>}
    """
    async with open_store() as db:
        await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await db.commit()

    return {"status": "deleted", "id": task_id}


if __name__ == "__main__":
    mcp.run()

from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

CHAT_DB = DATA_DIR / "chat.db"
MEMORY_DB = DATA_DIR / "memory.db"

USER_ID = "user_1001"
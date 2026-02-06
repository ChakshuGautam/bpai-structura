from typing import AsyncGenerator
from core.db.client import get_connection, put_connection


async def get_db() -> AsyncGenerator:
    """Yield a psycopg2 connection, then return it to the pool."""
    conn = get_connection()
    try:
        yield conn
    finally:
        put_connection(conn)

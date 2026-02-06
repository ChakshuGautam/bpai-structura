import os
import logging
import psycopg2
from psycopg2 import pool
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def _get_pool() -> pool.ThreadedConnectionPool:
    """Get or create the connection pool."""
    global _connection_pool
    if _connection_pool is None or _connection_pool.closed:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise ConnectionError(
                "DATABASE_URL not set. Database functionality will be unavailable."
            )
        try:
            _connection_pool = pool.ThreadedConnectionPool(
                minconn=2,
                maxconn=10,
                dsn=database_url,
            )
            logger.info("PostgreSQL connection pool initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL pool: {e}")
            raise
    return _connection_pool


def get_connection():
    """Get a connection from the pool."""
    return _get_pool().getconn()


def put_connection(conn):
    """Return a connection to the pool."""
    p = _get_pool()
    p.putconn(conn)


# Sentinel kept so existing imports of `supabase` don't break at import-time.
# The override repositories never reference this; it exists purely so that
# `from core.db.client import supabase` in un-overridden files won't crash.
supabase = None

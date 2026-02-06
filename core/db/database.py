from typing import AsyncGenerator
from supabase import Client

from .client import supabase

async def get_db() -> AsyncGenerator[Client, None]:
    """Get Supabase client instance."""
    try:
        yield supabase
    finally:
        # No need to close connection with Supabase client
        pass 
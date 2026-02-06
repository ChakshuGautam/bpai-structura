import os
from supabase import create_client, Client
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize supabase client as None
supabase: Optional[Client] = None

def initialize_supabase():
    """Initialize the Supabase client with proper error handling."""
    global supabase
    
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        logger.warning(
            "⚠️ Supabase URL or key not found in environment variables. "
            "Database functionality will be unavailable. "
            "Set SUPABASE_URL and SUPABASE_KEY environment variables if database access is needed."
        )
        return None
    
    try:
        supabase = create_client(url, key)
        logger.info("✅ Supabase client initialized successfully")
        return supabase
    except Exception as e:
        logger.error(f"❌ Failed to initialize Supabase client: {str(e)}")
        return None

# Try to initialize on import
initialize_supabase()

def check_tables_exist():
    """Check if required tables exist and print diagnostic information."""
    if not supabase:
        logger.warning("⚠️ Supabase client not initialized, can't check tables")
        return False
        
    try:
        # Using Supabase client to check if the table exists
        result = supabase.table("document_types").select("count", count="exact").execute()
        
        if result:
            count = result.count
            logger.info(f"✅ document_types table exists and has {count} records")
            return True
        else:
            logger.warning("❌ document_types table does not exist")
            logger.warning("Please run the migrations to create the necessary tables")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error checking tables: {str(e)}")
        logger.error("Please ensure the database connection is configured correctly")
        return False

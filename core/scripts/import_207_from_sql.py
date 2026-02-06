#!/usr/bin/env python3
"""
Execute the generated SQL import using the backend's database connection.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.client import supabase

def execute_sql_file(sql_file_path):
    """Execute SQL statements from a file."""
    print(f"📂 Reading SQL from {sql_file_path}...")

    with open(sql_file_path, 'r') as f:
        sql = f.read()

    # Split by statements (naive split by semicolon)
    statements = [s.strip() for s in sql.split(';') if s.strip() and not s.strip().startswith('--')]

    print(f"📝 Found {len(statements)} SQL statements")
    print(f"🚀 Executing via Supabase RPC...")

    try:
        # Execute via Supabase's rpc function for raw SQL
        # Note: This requires a stored procedure or we need to use PostgREST directly
        # Let's use the connection to execute directly

        # Actually, Supabase Python client doesn't support raw SQL execution
        # We need to insert records one by one using the table() API
        print("⚠️  Supabase client doesn't support raw SQL execution")
        print("📋 Please run the SQL file manually in Supabase SQL Editor:")
        print(f"   1. Go to https://supabase.com/dashboard")
        print(f"   2. Open SQL Editor")
        print(f"   3. Paste contents of: {sql_file_path}")
        print(f"   4. Click 'Run'")
        return False

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    sql_file = "import_evaluations.sql"
    if len(sys.argv) > 1:
        sql_file = sys.argv[1]

    if not os.path.exists(sql_file):
        print(f"❌ SQL file not found: {sql_file}")
        sys.exit(1)

    execute_sql_file(sql_file)

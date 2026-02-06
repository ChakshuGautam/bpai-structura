# Database Setup Guide for Supabase

## Overview

This document outlines the steps to set up the database for the DocumentType API. The application uses Supabase for data storage.

> **IMPORTANT**: Database schema management (tables, indexes, etc.) should ONLY be done through the Supabase dashboard. The application itself is designed to perform only data CRUD operations (Create, Read, Update, Delete) and should never modify the database structure.

## Prerequisites

- Supabase account and project
- Supabase URL and API key (stored in the SUPABASE_URL and SUPABASE_KEY environment variables)

## Database Schema Setup (Supabase Dashboard Only)

The application requires you to create the following table structure using the Supabase dashboard:

### document_types

This table stores document type definitions and schemas.

```sql
CREATE TABLE IF NOT EXISTS document_types (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    type VARCHAR(255) NOT NULL UNIQUE,
    description TEXT NOT NULL,
    schema JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add index for type lookup
CREATE INDEX IF NOT EXISTS idx_document_types_type ON document_types(type);

-- Create trigger to update updated_at on change
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS set_document_types_updated_at ON document_types;

CREATE TRIGGER set_document_types_updated_at
BEFORE UPDATE ON document_types
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();
```

### parsed_documents

This table stores parsed data from documents with their metadata.

```sql
CREATE TABLE IF NOT EXISTS parsed_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL,
    document_type_id UUID NOT NULL REFERENCES document_types(id) ON DELETE CASCADE,
    data JSONB NOT NULL,
    original_file_name TEXT NOT NULL,
    original_file_url TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes for common queries
CREATE INDEX IF NOT EXISTS idx_parsed_documents_user_id ON parsed_documents(user_id);
CREATE INDEX IF NOT EXISTS idx_parsed_documents_document_type_id ON parsed_documents(document_type_id);

-- Create trigger to update updated_at on change
CREATE TRIGGER set_parsed_documents_updated_at
BEFORE UPDATE ON parsed_documents
FOR EACH ROW
EXECUTE FUNCTION update_modified_column();

-- Add unique constraint to prevent duplicate documents
CREATE UNIQUE INDEX IF NOT EXISTS idx_parsed_documents_unique_doc 
ON parsed_documents(user_id, document_type_id, original_file_name);
```

## Recommended Setup Process

### Using the SQL Editor in Supabase (Recommended)

1. Log in to your Supabase dashboard
2. Navigate to the "SQL Editor" section
3. Create a new query
4. Copy and paste the SQL from above
5. Execute the SQL statements
6. Verify the table was created in the "Table Editor" section

### Alternative: Using Supabase Table Editor

1. Log in to your Supabase dashboard
2. Navigate to the "Table Editor" section
3. Click "New Table"
4. Create the table with the following columns:
   - id (uuid, primary key, default: gen_random_uuid())
   - type (varchar, unique, not null)
   - description (text, not null)
   - schema (jsonb, not null)
   - created_at (timestamptz, default: now())
   - updated_at (timestamptz, default: now())
5. After creating the table, add the index on the "type" column
6. Create and apply the trigger function for updating the "updated_at" column

## Verifying Connection

After setting up the table in Supabase, you can verify that the application can connect to the database:

```bash
# From the project root
PYTHONPATH=. python -c "from core.db.client import check_tables_exist; check_tables_exist()"
```

## Future Schema Changes

All future schema changes (adding columns, indexes, etc.) should be done directly through the Supabase dashboard. The application is designed to adapt to the schema it finds, but will not modify the schema itself.

## Troubleshooting

### Connection Issues

If you're experiencing connection issues:

1. Verify that the SUPABASE_URL and SUPABASE_KEY environment variables are correctly set
2. Check if your Supabase project is active
3. Ensure that your API key has the necessary permissions
4. Check if you're using the correct project URL

### Common Error Messages

- "Invalid API key": The Supabase API key is incorrect or has expired
- "Not found": The specified table doesn't exist
- "Invalid token": The authorization token is invalid or expired

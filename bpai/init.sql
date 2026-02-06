-- BPAI PostgreSQL schema
-- Matches existing Pydantic models from core/models/

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- document_types table
CREATE TABLE IF NOT EXISTS document_types (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id     UUID NOT NULL,
    type        TEXT NOT NULL,
    description TEXT NOT NULL,
    schema      JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT document_types_user_id_type_key UNIQUE (user_id, type)
);

-- html_files table
CREATE TABLE IF NOT EXISTS html_files (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id         UUID NOT NULL,
    file_id         UUID NOT NULL,
    marker_output   JSONB NOT NULL DEFAULT '{}',
    marker_version  TEXT NOT NULL DEFAULT '1.0',
    metadata        JSONB,
    status          TEXT NOT NULL DEFAULT 'pending',
    task_id         UUID,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_html_files_user_file ON html_files (user_id, file_id);
CREATE INDEX IF NOT EXISTS idx_html_files_task_id ON html_files (task_id);
CREATE INDEX IF NOT EXISTS idx_html_files_file_id ON html_files (file_id);

-- parsed_documents table
CREATE TABLE IF NOT EXISTS parsed_documents (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id             UUID NOT NULL,
    document_type_id    UUID NOT NULL REFERENCES document_types(id),
    data                JSONB NOT NULL DEFAULT '{}',
    original_file_name  TEXT NOT NULL,
    original_file_url   TEXT,
    file_id             UUID,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_parsed_documents_user ON parsed_documents (user_id);
CREATE INDEX IF NOT EXISTS idx_parsed_documents_doctype ON parsed_documents (document_type_id);
CREATE INDEX IF NOT EXISTS idx_parsed_documents_doctype_user_file ON parsed_documents (document_type_id, user_id, original_file_name);

-- Auto-update updated_at on document_types
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_document_types_updated_at
    BEFORE UPDATE ON document_types
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_html_files_updated_at
    BEFORE UPDATE ON html_files
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
